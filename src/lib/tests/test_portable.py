# -*- coding: UTF-8 -*-
""" Tests for pbox.portable: the dependency-free code that the tools exported from trained models embed.

- pbox.portable must only import the standard library (and itself) ;
- its primitives must give the same values as the implementations pbox uses (bintropy, pbox's entropy, tinyscript) ;
- its PE parser must follow LIEF's parser (the default parser of pbox) ; the rules it ports are checked on small PE
   files built here, and compared with LIEF itself when it is installed, then on the PE samples of the repository.

The loaded package is 'portable' (from src/pbox), not 'pbox.portable', to check that it works without pbox.
"""
import ast
import random
import struct
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1].joinpath("src")
PORTABLE = SRC.joinpath("pbox", "portable")
REPO = Path(__file__).resolve().parents[3]
SAMPLES = sorted(p for d in ("tests/samples", "src/files/packers") for p in REPO.joinpath(d).rglob("*")
                 if p.is_file() and p.read_bytes()[:2] == b"MZ")

sys.path.insert(0, str(SRC))                    # pbox from the source tree (for test_pbox_parser)
sys.path.insert(0, str(SRC.joinpath("pbox")))   # pbox/portable as the top-level package 'portable'
from portable.primitives import block_entropy, entropy, zeropad  # noqa: E402
from portable.parsers.pe import PE, NullSection, PEError          # noqa: E402


# ------------------------------------------------------------------------------------------------------ standalone
def test_only_standard_library_imports():
    allowed = set(sys.stdlib_module_names)
    for py in PORTABLE.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:   # relative import: within pbox.portable only
                    depth = len(py.relative_to(PORTABLE).parts) - 1
                    assert node.level <= depth + 1, f"{py.name}: relative import out of pbox.portable"
                    continue
                names = [node.module]
            else:
                continue
            for n in names:
                assert n.split(".")[0] in allowed, f"{py.name} imports '{n}', which is not in the standard library"


# ------------------------------------------------------------------------------------------------------ primitives
def _inputs(n=200, seed=0):
    rnd = random.Random(seed)
    yield from [b"", b"\x00", b"\x00" * 300, bytes(range(256)) * 3, b"A" * 257]
    for _ in range(n):
        yield bytes(rnd.choice([0, 0, 0, 1, 2, 255, rnd.randrange(256)]) for _ in range(rnd.randrange(1, 3000)))


def test_block_entropy_as_bintropy():
    bintropy = pytest.importorskip("bintropy")
    for d in _inputs():
        for args in [(0,), (256,), (512,), (256, True), (256, False, True), (256, True, True), (1024, False, True)]:
            assert block_entropy(d, *args) == bintropy.entropy(d, *args), (len(d), args)


def test_entropy_as_pbox():
    utils = pytest.importorskip("exeplot.utils")
    from math import log2
    def reference(data):   # pbox.helpers.utils.entropy(data), without importing pbox
        e, t = 0., len(data)
        for _, n in utils.ngrams_distribution(data, 1, 1, 256):
            e -= (p := n / t) * log2(p or 1)
        return e
    for d in _inputs():
        if d:
            assert entropy(d) == reference(d), len(d)


def test_zeropad_as_tinyscript():
    helpers = pytest.importorskip("tinyscript.helpers")
    rnd = random.Random(1)
    for _ in range(500):
        lst = [rnd.randrange(256) for _ in range(rnd.choice([0, 1, 5, 63, 64, 65, 200]))]
        for n in (0, 1, 5, 64):
            assert zeropad(n)(list(lst)) == helpers.zeropad(n)(list(lst))


# ------------------------------------------------------------------------------------------- synthetic PE files
def build_pe(sections=(), dirs=None, pe32plus=False, file_alignment=0x200, section_alignment=0x1000, entrypoint=0x1000,
             overlay=b"", number_of_dirs=16, extra=None):
    """ Build a minimal PE file.

    :param sections: (name bytes, virtual address, virtual size, raw offset, raw size, characteristics, data)
    :param dirs:     {data directory index: (rva, size)}
    :param extra:    {file offset: bytes} written after everything else (e.g. strings referenced by RVAs that are
                      outside every section, hence taken as file offsets by LIEF)
    """
    opt_size = (112 if pe32plus else 96) + 8 * number_of_dirs
    table = 0x40 + 4 + 20 + opt_size
    headers_size = table + 40 * len(sections)
    sizeof_headers = max(0x200, _align(headers_size, file_alignment))
    end = max([sizeof_headers] + [o + len(d) for _, _, _, o, _, _, d in sections])
    out = bytearray(end)
    out[0:2] = b"MZ"
    struct.pack_into("<I", out, 0x3c, 0x40)
    out[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", out, 0x44, 0x8664 if pe32plus else 0x14c, len(sections), 0, 0, 0, opt_size, 0x102)
    opt = 0x58
    struct.pack_into("<H", out, opt, 0x20b if pe32plus else 0x10b)
    struct.pack_into("<I", out, opt + 16, entrypoint)
    if pe32plus:
        struct.pack_into("<Q", out, opt + 24, 0x140000000)
    else:
        struct.pack_into("<I", out, opt + 28, 0x400000)
    struct.pack_into("<II", out, opt + 32, section_alignment, file_alignment)
    struct.pack_into("<II", out, opt + 56, 0x100000, sizeof_headers)
    struct.pack_into("<H", out, opt + 68, 2)   # subsystem: GUI
    struct.pack_into("<I", out, opt + (108 if pe32plus else 92), number_of_dirs)
    for i, (rva, size) in (dirs or {}).items():
        struct.pack_into("<II", out, opt + (112 if pe32plus else 96) + 8 * i, rva, size)
    for i, (name, va, vs, off, rs, ch, data) in enumerate(sections):
        struct.pack_into("<8sIIIIIIHHI", out, table + 40 * i, name, vs, va, rs, off, 0, 0, 0, 0, ch)
        out[off:off+len(data)] = data
    for off, data in (extra or {}).items():
        if off + len(data) > len(out):
            out.extend(b"\0" * (off + len(data) - len(out)))
        out[off:off+len(data)] = data
    return bytes(out) + overlay


def _align(v, a):
    return v + (a - v % a) % a


def _import_section(va, dlls, pe32plus=False, dir_count=None):
    """ Data of a section holding an import table at its start: dlls = [(name, [entry names or ordinals])]. """
    ptr = 8 if pe32plus else 4
    n = len(dlls) if dir_count is None else dir_count
    desc_size, body = 20 * (len(dlls) + 1), bytearray()
    strings_at = desc_size + sum((len(e) + 1) * ptr * 2 for _, e in dlls)
    strings, descs, tables = bytearray(), bytearray(), bytearray()
    tables_at = desc_size
    for name, entries in dlls:
        name_rva = va + strings_at + len(strings)
        strings += name + b"\0"
        thunks = []
        for e in entries:
            if isinstance(e, int):
                thunks.append(e | (1 << (63 if pe32plus else 31)))
            else:
                thunks.append(va + strings_at + len(strings))
                strings += b"\0\0" + e + b"\0"
        ilt_rva = va + tables_at + len(tables)
        tables += b"".join(struct.pack("<Q" if pe32plus else "<I", t) for t in thunks + [0])
        iat_rva = va + tables_at + len(tables)
        tables += b"".join(struct.pack("<Q" if pe32plus else "<I", t) for t in thunks + [0])
        descs += struct.pack("<5I", ilt_rva, 0, 0, name_rva, iat_rva)
    body = descs + b"\0" * 20 + tables + strings
    return bytes(body), (va, 20 * n), (va + desc_size, 8)


CODE = 0x60000020
DATA = 0xC0000040


def _lief_view(data):
    """ The properties compared between LIEF and the portable parser (see _portable_view). """
    lief = pytest.importorskip("lief")
    lief.logging.disable()
    b = lief.PE.parse(list(data))
    eps = b.section_from_rva(b.optional_header.addressof_entrypoint)
    return {
        'names': [s.name for s in b.sections], 'content': [bytes(s.content) for s in b.sections],
        'overlay': bytes(b.overlay), 'ep_section': None if eps is None else eps.name,
        'imports': [(i.name, [e.name for e in i.entries]) for i in b.imports],
        'has_imports': b.has_imports, 'has_exports': b.has_exports, 'has_tls': b.has_tls,
        'has_delay_imports': b.has_delay_imports, 'exported': [f.name for f in b.exported_functions],
    }


def _portable_view(data):
    p = PE(data)
    eps = p.entrypoint_section
    return {
        'names': [s.name for s in p], 'content': [s.content for s in p], 'overlay': p.overlay,
        'ep_section': None if isinstance(eps, NullSection) else eps.name, 'imports': p.imports,
        'has_imports': p.has_imports, 'has_exports': p.has_exports, 'has_tls': p.has_tls,
        'has_delay_imports': p.has_delay_imports, 'exported': p.exported_functions,
    }


def _check(data, **expected):
    """ Check the portable parser's values, and that LIEF agrees when it is installed. """
    view = _portable_view(data)
    for k, v in expected.items():
        assert view[k] == v, k
    try:
        import lief  # noqa: F401
    except ImportError:
        return view
    assert _lief_view(data) == view
    return view


@pytest.mark.parametrize("pe32plus", [False, True])
def test_sections_and_overlay(pe32plus):
    text = (b".text\0\0\0", 0x1000, 0x80, 0x200, 0x200, CODE, bytes(range(256)) * 2)
    data = (b".data\0\0\0", 0x2000, 0x1000, 0x400, 0x200, DATA, b"\x01" * 0x200)
    _check(build_pe([text, data], pe32plus=pe32plus, overlay=b"OVERLAY"),
           names=[".text", ".data"],
           content=[bytes(range(256))[:0x80], b"\x01" * 0x200],   # limited to the virtual size when smaller
           overlay=b"OVERLAY", ep_section=".text", has_imports=False, has_exports=False)


def test_section_names_bytes_and_empty():
    s1 = (b"\xffAB\0junk", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    s2 = (b"\0\x10@\0\0\0\0\0", 0x2000, 0x200, 0x400, 0x200, DATA, b"\0" * 0x200)
    view = _check(build_pe([s1, s2]), names=[b"\xffAB", ""])
    p = PE(build_pe([s1, s2]))
    assert [s.real_name for s in p] == ["\xffAB", ""]   # bytes names decoded as Latin-1, as pbox does
    assert len(p.empty_name_sections) == 1


def test_no_section_whole_file_is_overlay():
    data = build_pe([], entrypoint=0x1000)
    view = _check(data, names=[], overlay=data, ep_section=None)
    with pytest.raises(PEError):
        PE(data).bytes_after_entrypoint(64)
    assert PE(data).entrypoint_section.entropy == 0.


def test_entrypoint_outside_sections():
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    p = PE(_check(build_pe([text], entrypoint=0x9000), ep_section=None) and build_pe([text], entrypoint=0x9000))
    assert isinstance(p.entrypoint_section, NullSection) and p.entrypoint_section.is_writable is True   # as pbox: flags of -1


@pytest.mark.parametrize("pe32plus", [False, True])
def test_imports(pe32plus):
    body, imp_dir, iat_dir = _import_section(0x2000, [(b"KERNEL32.dll", [b"LoadLibraryA", b"GetProcAddress", 7]),
                                                      (b"USER32.dll", [b"MessageBoxA"])], pe32plus)
    idata = (b".idata\0\0", 0x2000, 0x1000, 0x400, _align(len(body), 0x200), DATA, body)
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    p = PE(build_pe([text, idata], {1: imp_dir, 12: iat_dir}, pe32plus=pe32plus))
    _check(build_pe([text, idata], {1: imp_dir, 12: iat_dir}, pe32plus=pe32plus),
           imports=[("KERNEL32.dll", ["LoadLibraryA", "GetProcAddress", ""]), ("USER32.dll", ["MessageBoxA"])],
           has_imports=True)
    assert p.imported_dlls == {"kernel32.dll", "user32.dll"} and len(p.imported_apis) == 4


def test_imports_limited_to_directory_size():
    """ Only the descriptors within the declared size of the import directory are read. """
    body, imp_dir, iat_dir = _import_section(0x2000, [(b"KERNEL32.dll", [b"LoadLibraryA"]),
                                                      (b"USER32.dll", [b"MessageBoxA"])], dir_count=1)
    idata = (b".idata\0\0", 0x2000, 0x1000, 0x400, _align(len(body), 0x200), DATA, body)
    _check(build_pe([idata], {1: imp_dir, 12: iat_dir}, entrypoint=0x2000), imports=[("KERNEL32.dll", ["LoadLibraryA"])])


def test_imports_invalid_names():
    """ Too short or unprintable DLL names are skipped ; unprintable function names are discarded. """
    body, imp_dir, iat_dir = _import_section(0x2000, [(b"K32", [b"LoadLibraryA"]), (b"KERNEL32.dll", [b"Bad\x01"]),
                                                      (b"USER32.dll", [b"MessageBoxA"])])
    idata = (b".idata\0\0", 0x2000, 0x1000, 0x400, _align(len(body), 0x200), DATA, body)
    _check(build_pe([idata], {1: imp_dir, 12: iat_dir}, entrypoint=0x2000),
           imports=[("KERNEL32.dll", []), ("USER32.dll", ["MessageBoxA"])])


def test_imports_need_iat_directory():
    """ Without an IAT data directory entry (fewer than 13 directories), LIEF parses no import. """
    body, imp_dir, _ = _import_section(0x2000, [(b"KERNEL32.dll", [b"LoadLibraryA"])])
    idata = (b".idata\0\0", 0x2000, 0x1000, 0x400, _align(len(body), 0x200), DATA, body)
    _check(build_pe([idata], {1: imp_dir}, entrypoint=0x2000, number_of_dirs=10), has_imports=False)


def test_rva_to_offset_aligns_up():
    """ LIEF aligns the raw data pointer up to the file alignment: a table in a section whose raw data starts at an
         unaligned offset is read from another place (here, as in FSG-packed files, the import table is not found). """
    body, imp_dir, iat_dir = _import_section(0x2000, [(b"KERNEL32.dll", [b"LoadLibraryA"])])
    idata = (b".idata\0\0", 0x2000, 0x1000, 0x300, _align(len(body), 0x100), DATA, body)
    p = PE(build_pe([idata], {1: imp_dir, 12: iat_dir}, entrypoint=0x2000, file_alignment=0x1000))
    assert p.rva_to_offset(0x2010) == 0x1010 and p.imports == []
    _check(build_pe([idata], {1: imp_dir, 12: iat_dir}, entrypoint=0x2000, file_alignment=0x1000), has_imports=False)


def _export_section(va, names, n_funcs, functions):
    """ Export directory at the start of a section: names = [(name, function index)], functions = {index: rva}. """
    hdr_size, n = 40, len(names)
    funcs_at, names_at = hdr_size, hdr_size + 4 * n_funcs
    ords_at = names_at + 4 * n
    strings_at = ords_at + 2 * n
    strings, name_rvas = bytearray(b"test.dll\0"), []
    for name, _ in names:
        name_rvas.append(va + strings_at + len(strings))
        strings += name + b"\0"
    body = struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, va + strings_at, 1, n_funcs, n, va + funcs_at, va + names_at,
                       va + ords_at)
    body += b"".join(struct.pack("<I", functions.get(i, 0)) for i in range(n_funcs))
    body += b"".join(struct.pack("<I", r) for r in name_rvas)
    body += b"".join(struct.pack("<H", i) for _, i in names)
    return body + strings


def test_exports_named_entries():
    """ The exported functions are the export entries with a name and an address. """
    body = _export_section(0x2000, [(b"f0", 0), (b"f2", 2), (b"nul", 3)], 5, {0: 0x1000, 1: 0x1010, 2: 0x1020})
    edata = (b".edata\0\0", 0x2000, 0x1000, 0x400, _align(len(body), 0x200), DATA, body)
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    _check(build_pe([text, edata], {0: (0x2000, len(body))}), has_exports=True, exported=["f0", "f2"])


def test_exports_unreadable_or_corrupted():
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    # export directory beyond the end of the file
    _check(build_pe([text], {0: (0x7000000, 40)}), has_exports=False, exported=[])
    # too many entries: corrupted, hence no export
    body = struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, 0, 1, 0x2000000, 0, 0x2100, 0, 0)
    edata = (b".edata\0\0", 0x2000, 0x1000, 0x400, 0x200, DATA, body)
    _check(build_pe([text, edata], {0: (0x2000, 40)}), has_exports=False)


def test_delay_imports_and_tls():
    desc = struct.pack("<8I", 1, 0x2100, 0, 0, 0, 0, 0, 0)
    body = desc + b"\0" * 32 + b"\0" * (0x100 - 64) + b"ADVAPI32.dll\0"
    didata = (b".didat\0\0", 0x2000, 0x1000, 0x400, 0x200, DATA, body)
    _check(build_pe([didata], {13: (0x2000, 64)}, entrypoint=0x2000), has_delay_imports=True)
    # directory larger than the content of its section: no delay import
    _check(build_pe([didata], {13: (0x2000, 0x2000)}, entrypoint=0x2000), has_delay_imports=False)
    # directory in a section without raw data: no delay import
    bss = (b".didat\0\0", 0x2000, 0x1000, 0, 0, DATA, b"")
    _check(build_pe([bss], {13: (0x2000, 64)}, entrypoint=0x2000), has_delay_imports=False)
    # TLS directory: present as soon as its structure can be read
    _check(build_pe([didata], {9: (0x2000, 24)}, entrypoint=0x2000), has_tls=True)
    _check(build_pe([didata], {9: (0x7000000, 24)}, entrypoint=0x2000), has_tls=False)


# ------------------------------------------------------------------------------------------------- real samples
@pytest.mark.skipif(len(SAMPLES) == 0, reason="no PE sample in the repository")
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_samples_as_lief(sample):
    lief = pytest.importorskip("lief")
    data = sample.read_bytes()
    lief.logging.disable()
    if lief.PE.parse(list(data)) is None:
        pytest.skip("not parsed by LIEF")
    assert _portable_view(data) == _lief_view(data)
    p = PE(data)
    assert len(p.imported_apis) >= 0 and p.entropy == block_entropy(data)


def test_pbox_parser():
    """ pbox's 'portable' parser gives the portable parser's values. """
    pytest.importorskip("pbox")
    sample = next((s for s in SAMPLES if "upx_calc" in s.name), None)
    if sample is None:
        pytest.skip("sample not found")
    from pbox import Executable
    p = Executable(str(sample)).parse("portable")
    ref = PE(sample.read_bytes())
    assert p.parser == "portable"
    assert p['entropy'] == ref.entropy and p['block_entropy'](256) == ref.block_entropy(256)
    assert p['average_block_entropy_per_section'](256) == ref.average_block_entropy_per_section(256)
    assert p['bytes_after_entrypoint'](64) == ref.bytes_after_entrypoint(64)
    assert p['entrypoint_section']['entropy'] == ref.entrypoint_section.entropy
    assert p['imported_apis'] == ref.imported_apis and p['has_tls'] == ref.has_tls
    assert [s.name for s in p] == [s.name for s in ref]
    # section names from pbox's data (standard sections, packer sections) are set from the executable's format
    assert len(p['known_packer_sections']) > 0   # UPX0, UPX1


# ------------------------------------------------------------------------------------------------------- pefeats
from portable.extractors.pefeats import FEATURES, PefeatsError, pefeats, pefeats_values  # noqa: E402

EXPECTED = Path(__file__).resolve().parent.joinpath("data", "pefeats_expected.json")


def test_pefeats_features_count():
    assert len(FEATURES) == 119 and len(set(FEATURES)) == 119


@pytest.mark.parametrize("sample", sorted(__import__("json").loads(EXPECTED.read_text())), ids=lambda s: s.split("/")[-1])
def test_pefeats_as_native_output(sample):
    """ Same output as pefeats built from https://github.com/packing-box/pefeats (recorded in data/). """
    expected = __import__("json").loads(EXPECTED.read_text())[sample]
    assert pefeats(REPO.joinpath(sample).read_bytes()) == expected


@pytest.mark.skipif(not __import__("os").environ.get("PEFEATS"), reason="set PEFEATS to a pefeats executable")
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.name)
def test_pefeats_as_native_live(sample):
    import os, subprocess
    out = subprocess.run([os.environ['PEFEATS'], str(sample)], capture_output=True, text=True)
    assert out.returncode == 0
    assert pefeats(sample.read_bytes()) == out.stdout.strip().split(",")[-119:]


def test_pefeats_values_as_pbox_parses_them():
    sample = next(s for s in SAMPLES if s.name == "calc.exe")
    v = pefeats_values(sample.read_bytes())
    assert all(isinstance(x, (int, float)) for x in v) and isinstance(v[23], float) and isinstance(v[21], int)


def test_pefeats_failures():
    # no section: pefeats exits with code 3
    with pytest.raises(PefeatsError):
        pefeats(build_pe([]))
    # not a PE file
    with pytest.raises(PefeatsError):
        pefeats(b"ELF" + b"\0" * 100)
    # entry point in its valid range but in no section: pefeats reads out of bounds, not reproducible
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    with pytest.raises(PefeatsError):
        pefeats(build_pe([text], entrypoint=0x9000))


def test_pefeats_no_import_directory():
    """ When the import directory cannot be read, the 4 import values are -1 (the ratio without decimals, as pefeats
         prints it with another output stream). """
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    v = pefeats(build_pe([text]))
    assert v[112:116] == ["-1", "-1", "-1", "-1"] and v[116] == "-1" and v[117] == "1" and v[118] == "-1"
    assert v[:8] == ["0"] * 8 and v[21:23] == ["1", "0"] and v[23] == "1.000"


def test_pefeats_imports():
    body, imp_dir, iat_dir = _import_section(0x2000, [(b"KERNEL32.dll", [b"LoadLibraryA", b"GetProcAddress", 7]),
                                                      (b"USER32.dll", [b"MessageBoxW"])])
    idata = (b".idata\0\0", 0x2000, 0x1000, 0x400, _align(len(body), 0x200), DATA, body)
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    v = pefeats(build_pe([text, idata], {1: imp_dir, 12: iat_dir}))
    assert v[112:116] == ["2", "4", "2", "0.500"]


def test_pefeats_64bit_layout_depends_on_machine():
    """ PeLib reads the 64-bit layout only for AMD64/IA64 machines with the PE32+ magic. """
    text = (b".text\0\0\0", 0x1000, 0x200, 0x200, 0x200, CODE, b"\x90" * 0x200)
    data = bytearray(build_pe([text], pe32plus=True))
    assert pefeats(bytes(data))[9] == str(0x140000000)     # image base (64 bits)
    data[0x44:0x46] = b"\x4c\x01"                           # i386 machine with the PE32+ magic: 32-bit layout
    assert pefeats(bytes(data))[9] != str(0x140000000)
