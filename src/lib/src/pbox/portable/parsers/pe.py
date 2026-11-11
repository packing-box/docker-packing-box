# -*- coding: UTF-8 -*-
""" Minimal PE parser, with no dependency outside the standard library.

It parses what the feature expressions use (headers, sections, entry point, imports, delay imports, exports, TLS) and
 exposes the same properties as pbox's parsed binaries (see pbox.core.executable.parsers), with the semantics of the
 LIEF-based parser that pbox uses by default, so that features computed with either parser are identical.

The parsing of the headers, sections and directories follows LIEF 1.0.0 (src/PE/Parser.cpp, Parser.tcc, Binary.cpp),
 including how it handles malformed executables (common with packed ones): how relative virtual addresses are turned
 into file offsets, which entries are discarded, where parsing stops, how strings are read. The functions that port a
 LIEF routine name it in their docstring.

Items are reachable both as attributes and as keys (binary.entropy or binary['entropy']), as in the YAML expressions.
"""
import struct

from ..primitives import block_entropy, entropy, zeropad


__all__ = ["PE", "PEError", "Section"]

# section characteristics
SCN_CNT_CODE, SCN_CNT_INITIALIZED_DATA, SCN_CNT_UNINITIALIZED_DATA = 0x20, 0x40, 0x80
SCN_MEM_DISCARDABLE, SCN_MEM_SHARED, SCN_MEM_EXECUTE, SCN_MEM_READ, SCN_MEM_WRITE = \
    0x02000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000
# data directories
DIR_EXPORT, DIR_IMPORT, DIR_TLS, DIR_IAT, DIR_DELAY_IMPORT = 0, 1, 9, 12, 13
# limits of LIEF's parser (include/LIEF/PE/Parser.hpp, src/PE/Parser.*)
MAX_SECTIONS, MAX_DATA_DIRECTORIES = 1000, 30
MAX_IMPORT_ENTRIES, MAX_IMPORT_NAME_SIZE, MAX_DLL_NAME_SIZE, MIN_DLL_NAME_SIZE = 0x10000, 0x1000, 255, 4
EXPORT_ENTRIES_LIMIT, MAX_EXPORT_NAME_SIZE = 0x1000000, 4096
_U64 = (1 << 64) - 1


class PEError(ValueError):
    pass


def _is_printable(s):
    """ LIEF's is_printable: every character between ' ' and '~'. """
    return all(0x20 <= c <= 0x7e for c in s)


def _is_valid_dll_name(name):
    """ LIEF's Parser::is_valid_dll_name. """
    return MIN_DLL_NAME_SIZE <= len(name) <= MAX_DLL_NAME_SIZE and _is_printable(name)


def _is_valid_import_name(name):
    """ LIEF's Parser::is_valid_import_name (C's isprint: from ' ' to '~'). """
    return 0 < len(name) <= MAX_IMPORT_NAME_SIZE and _is_printable(name)


def _align(value, align_on):
    """ LIEF's align: round up to a multiple of align_on. """
    if align_on == 0:
        return value
    r = value % align_on
    return value + (align_on - r) if r > 0 else value


def _str(raw):
    """ Bytes as LIEF's Python bindings give strings: str if valid UTF-8, otherwise bytes. """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw


class _Item:
    """ Access to attributes as keys, as pbox's GetItemMixin does (keys with dots reach sub-attributes). """
    def __getitem__(self, name):
        if not isinstance(name, str) or name.startswith("_"):
            raise KeyError(name)
        obj = self
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj


class Section(_Item):
    """ PE section. """
    def __init__(self, binary, name, virtual_size, virtual_address, size, offset, characteristics, content):
        self.binary, self.name, self.characteristics = binary, name, characteristics
        self.virtual_size, self.virtual_address, self.size, self.offset = virtual_size, virtual_address, size, offset
        self.content = content

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"<Section {self.name!r}>"

    def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True):
        return block_entropy(self.bytes, blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)

    @property
    def bytes(self):
        return self.content or b""

    @property
    def entropy(self):
        return entropy(self.bytes or b" ")

    @property
    def flags(self):
        return self.characteristics

    @property
    def is_code(self):
        return self.characteristics & SCN_CNT_CODE > 0

    @property
    def is_data(self):
        return self.characteristics & (SCN_CNT_INITIALIZED_DATA | SCN_CNT_UNINITIALIZED_DATA) > 0

    @property
    def is_discardable(self):
        return self.characteristics & SCN_MEM_DISCARDABLE > 0

    @property
    def is_executable(self):
        return self.characteristics & SCN_MEM_EXECUTE > 0

    @property
    def is_readable(self):
        return self.characteristics & SCN_MEM_READ > 0

    @property
    def is_shared(self):
        return self.characteristics & SCN_MEM_SHARED > 0

    @property
    def is_writable(self):
        return self.characteristics & SCN_MEM_WRITE > 0

    @property
    def raw_data_size(self):
        return self.size

    @property
    def real_name(self):
        # as pbox: bytes names are decoded as UTF-8, or Latin-1 if they are not valid UTF-8 (tinyscript's ensure_str)
        n = self.name
        if isinstance(n, bytes):
            try:
                n = n.decode("utf-8")
            except UnicodeDecodeError:
                n = n.decode("latin-1")
        return self.binary.real_section_names.get(n, n).split("\0")[0]


class NullSection(_Item):
    """ Stands for a missing section (e.g. when the entry point is outside every section), with the values pbox's
         NullSection gives to the features: no name, no content, and -1 for the sizes, offsets and flags (attributes
         named *_address, *alignment, characteristics, flags, *offset, *size), hence every flag test is True. """
    name, content = None, None
    characteristics = flags = offset = size = virtual_address = virtual_size = raw_data_size = -1

    def __repr__(self):
        return "<NullSection>"

    def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True):
        return block_entropy(b"", blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)

    @property
    def bytes(self):
        return b""

    @property
    def entropy(self):
        return entropy(b" ")

    # flag tests on characteristics=-1 (all bits set), as with pbox's sections built on a NullSection
    is_code = is_data = is_discardable = is_executable = is_readable = is_shared = is_writable = True


class PE(_Item):
    """ Parsed PE executable.

    :param data:                        bytes of the executable, or path to it
    :param standard_section_names:      names of the standard sections (pbox's data: STANDARD_SECTION_NAMES)
    :param common_packer_section_names: names of sections commonly used by packers (COMMON_PACKER_SECTION_NAMES)
    """
    def __init__(self, data, standard_section_names=(), common_packer_section_names=()):
        if not isinstance(data, (bytes, bytearray)):
            with open(data, "rb") as f:
                data = f.read()
        self.data = bytes(data)
        self.standard_section_names = list(standard_section_names)
        self.common_packer_section_names = list(common_packer_section_names)
        self._cache = {}
        self._parse_headers()
        self._parse_sections()
        self._parse_data_directories()

    def __iter__(self):
        return iter(self.sections)

    def __len__(self):
        return len(self.sections)

    def __repr__(self):
        return f"<PE {'PE32+' if self.is_pe32plus else 'PE32'} ({len(self.sections)} sections)>"

    # ---------------------------------------------------------------------------------------------- stream reading
    def _peek(self, fmt, offset):
        """ Little-endian values at a file offset, or None if they are not entirely in the file (as LIEF's peek). """
        size = struct.calcsize("<" + fmt)
        if offset < 0 or offset + size > len(self.data):
            return None
        return struct.unpack_from("<" + fmt, self.data, offset)

    def _peek_string(self, offset, maxsize=None):
        """ LIEF's BinaryStream::peek_string_at: the bytes up to the first NUL ; when the end of the data (or maxsize)
             is reached before a NUL, the last byte read is dropped ; an empty string when offset is out of the data. """
        d = self.data
        if offset < 0 or offset >= len(d):
            return b""
        end = len(d) if maxsize is None else min(len(d), offset + maxsize)
        nul = d.find(b"\0", offset, end)
        return d[offset:nul] if nul >= 0 else d[offset:end-1]

    def rva_to_offset(self, rva):
        """ LIEF's Binary::rva_to_offset: the first section covering the RVA (up to the larger of its virtual and raw
             sizes) gives the offset, with its virtual address and raw data pointer aligned up (to the section and
             file alignments) ; outside every section, the RVA is taken as is. NB: may give an offset outside the file,
             or wrap around as in C (64 bits), e.g. when the aligned virtual address is above the RVA. """
        for s in self.sections:
            if s.virtual_address <= rva < s.virtual_address + max(s.virtual_size, s.size):
                sa, fa = self.section_alignment, self.file_alignment
                if sa < 0x1000:
                    sa = fa
                return (rva - _align(s.virtual_address, sa) + _align(s.offset, fa)) & _U64
        return rva

    def section_from_offset(self, offset):
        """ LIEF's Binary::section_from_offset: the first section whose raw data holds the offset. """
        for s in self.sections:
            if s.offset <= offset < s.offset + s.size:
                return s

    def section_from_rva(self, rva):
        """ LIEF's Binary::section_from_rva: the first section whose virtual size covers the RVA. """
        for s in self.sections:
            if s.virtual_address <= rva < s.virtual_address + s.virtual_size:
                return s

    # ---------------------------------------------------------------------------------------------- headers, sections
    def _parse_headers(self):
        d = self.data
        if len(d) < 0x40 or d[:2] != b"MZ":
            raise PEError("not a PE file (no MZ header)")
        self.pe_offset = struct.unpack_from("<I", d, 0x3c)[0]
        if d[self.pe_offset:self.pe_offset+4] != b"PE\0\0":
            raise PEError("not a PE file (no PE signature)")
        coff = self.pe_offset + 4
        if (h := self._peek("HHIIIHH", coff)) is None:
            raise PEError("truncated COFF header")
        (self.machine, self.number_of_sections, self.timedatestamp, self.pointerto_symbol_table,
         self.numberof_symbols, self.sizeof_optional_header, self.characteristics) = h
        self.optional_header_offset = opt = coff + 20
        if (m := self._peek("H", opt)) is None or m[0] not in (0x10b, 0x20b):
            raise PEError("bad optional header magic")
        self.magic = m[0]
        self.is_pe32plus = self.magic == 0x20b
        if (h := self._peek("112s" if self.is_pe32plus else "96s", opt)) is None:
            raise PEError("truncated optional header")
        self.addressof_entrypoint = self._peek("I", opt + 16)[0]
        self.imagebase = self._peek("Q", opt + 24)[0] if self.is_pe32plus else self._peek("I", opt + 28)[0]
        self.section_alignment, self.file_alignment = self._peek("II", opt + 32)
        self.sizeof_image, self.sizeof_headers = self._peek("II", opt + 56)
        self.numberof_rva_and_size = self._peek("I", opt + (108 if self.is_pe32plus else 92))[0]

    def _parse_sections(self):
        """ LIEF's Parser::parse_sections and read_section_content (for a file, not a memory view). """
        self.sections, d = [], self.data
        table = self.optional_header_offset + self.sizeof_optional_header
        for i in range(min(self.number_of_sections, MAX_SECTIONS)):
            if (raw := self._peek("8sIIIIIIHHI", table + 40 * i)) is None:
                break
            raw_name, vsize, vaddr, size, offset, *_, characteristics = raw
            # the name up to the first NUL, as a string if it is valid UTF-8, otherwise as bytes
            name = _str(raw_name.split(b"\0")[0])
            # the content: the raw data, limited to the virtual size when it is set ("according to Corkami"), and to
            #  the end of the file ; nothing if it starts after the end of the file
            n = min(vsize, size) if vsize > 0 else size
            content = d[offset:offset+n] if n > 0 and offset < len(d) else b""
            self.sections.append(Section(self, name, vsize, vaddr, size, offset, characteristics, content))

    def _parse_data_directories(self):
        """ LIEF's Parser::parse_data_directories: the directories are read from the end of the fixed part of the
             optional header (whatever its declared size) ; if one of them cannot be read, none of the import, export,
             TLS and delay-import tables is parsed. """
        start = self.optional_header_offset + (112 if self.is_pe32plus else 96)
        self.data_directories, self._dirs_ok = [], True
        for i in range(min(self.numberof_rva_and_size, MAX_DATA_DIRECTORIES)):
            if (dd := self._peek("II", start + 8 * i)) is None:
                self._dirs_ok = False
                break
            self.data_directories.append(dd)

    def _dir(self, index):
        """ (RVA, size) of a data directory if it exists and the directories were read, else None. """
        return self.data_directories[index] if self._dirs_ok and index < len(self.data_directories) else None

    def _cached(self, key, func):
        if key not in self._cache:
            self._cache[key] = func()
        return self._cache[key]

    # ------------------------------------------------------------------------------------------- imports/exports/TLS
    @property
    def imports(self):
        """ List of (DLL name, [entry names]), as LIEF's Parser::parse_import_table gives them ; entries imported by
             ordinal have an empty name. """
        return self._cached("imports", self._parse_imports)

    def _parse_imports(self):
        imp_dir, iat_dir = self._dir(DIR_IMPORT), self._dir(DIR_IAT)
        if imp_dir is None or iat_dir is None or imp_dir[0] == 0:
            return []
        ptr, ptr_size = ("Q", 8) if self.is_pe32plus else ("I", 4)
        ordinal_mask = 1 << (63 if self.is_pe32plus else 31)
        result, pos = [], self.rva_to_offset(imp_dir[0])
        end = pos + imp_dir[1]
        while pos < end:
            if (desc := self._peek("5I", pos)) is None:
                break
            pos += 20
            if not any(desc):
                break
            ilt_rva, _, _, name_rva, iat_rva = desc
            if name_rva == 0:
                break
            name = self._peek_string(self.rva_to_offset(name_rva))
            if not _is_valid_dll_name(name):
                continue
            lt_off = self.rva_to_offset(ilt_rva) if ilt_rva > 0 else 0
            iat_off = self.rva_to_offset(iat_rva) if iat_rva > 0 else 0
            iat = table = 0
            if iat_off > 0 and (v := self._peek(ptr, iat_off)) is not None:
                iat = table = v[0]
                iat_off += ptr_size
            if lt_off > 0 and (v := self._peek(ptr, lt_off)) is not None:
                table = v[0]
                lt_off += ptr_size
            entries, idx = [], 0
            while (table != 0 or iat != 0) and idx < MAX_IMPORT_ENTRIES:
                idx += 1
                value = table if table > 0 else iat
                # ordinal: bit 31 (63) set and bits 30-16 (62-16) cleared (LIEF's ImportEntry::is_ordinal)
                if value & ordinal_mask and (value & ~ordinal_mask) >> 16 == 0:
                    entries.append("")
                else:
                    entry = self._peek_string(self.rva_to_offset(value) + 2, MAX_IMPORT_NAME_SIZE)
                    if _is_valid_import_name(entry):
                        entries.append(entry.decode("ascii"))
                if iat_off > 0:
                    iat = v[0] if (v := self._peek(ptr, iat_off)) is not None else 0
                    iat_off += ptr_size
                else:
                    iat = 0
                if lt_off > 0:
                    table = v[0] if (v := self._peek(ptr, lt_off)) is not None else 0
                    lt_off += ptr_size
                else:
                    table = 0
            result.append((name.decode("ascii"), entries))
        return result

    @property
    def delay_imports(self):
        """ Names of the delay-imported DLLs (an empty name for an invalid one), as LIEF's Parser::parse_delay_imports
             finds them: the descriptors are read within the content of the section holding the directory. """
        def _parse():
            if (dd := self._dir(DIR_DELAY_IMPORT)) is None or dd[0] == 0 or dd[1] == 0:
                return []
            # DataDirectory::content(): the section is found by the file offset of the RVA
            s = self.section_from_offset(self.rva_to_offset(dd[0]))
            if s is None or len(s.content) == 0:
                return []
            rel = dd[0] - s.virtual_address
            if rel < 0 or rel >= len(s.content) or rel + dd[1] > len(s.content):
                return []
            span, r = s.content[rel:rel+dd[1]], []
            for o in range(0, len(span) - 31, 32):
                desc = struct.unpack_from("<8I", span, o)
                if not any(desc):
                    break
                name = self._peek_string(self.rva_to_offset(desc[1]))
                r.append(name.decode("ascii") if name and _is_valid_dll_name(name) else "")
            return r
        return self._cached("delay_imports", _parse)

    @property
    def exports(self):
        """ Export entries [(name, address)] as LIEF's Parser::parse_exports keeps them, or None if there is no export
             table (or a corrupted one). """
        return self._cached("exports", self._parse_exports)

    def _parse_exports(self):
        if (dd := self._dir(DIR_EXPORT)) is None or dd[0] == 0:
            return None
        rva, size = dd
        u32 = lambda x: x & 0xffffffff  # LIEF keeps these offsets in uint32_t variables
        if (hdr := self._peek("IIHHIIIIIII", u32(self.rva_to_offset(rva)))) is None:
            return None
        _, _, _, _, _, ordinal_base, n_addr, n_names, addr_rva, names_rva, ordinals_rva = hdr
        if n_addr > EXPORT_ENTRIES_LIMIT or n_names > EXPORT_ENTRIES_LIMIT:
            return None
        addr_off, ord_off, names_off = (u32(self.rva_to_offset(x)) for x in (addr_rva, ordinals_rva, names_rva))
        start, end = rva, u32(rva + size)
        entries, corrupted = [], set()   # entries: [ordinal, address, is_extern, name]
        for i in range(n_addr):
            if (v := self._peek("I", addr_off + 4 * i)) is None:
                break
            value, ordinal = v[0], (i + ordinal_base) & 0xffff
            is_extern = start <= value < end
            entry = [ordinal, 0 if is_extern else value, is_extern, b""]
            if value == 0:
                corrupted.add(ordinal)
            if is_extern and value > 0:
                name = self._peek_string(u32(self.rva_to_offset(value)))
                entry[3] = name if len(name) <= MAX_EXPORT_NAME_SIZE and _is_printable(name) else b""
            entries.append(entry)
        for i in range(n_names):
            if (v := self._peek("H", ord_off + 2 * i)) is None:
                break
            if v[0] >= len(entries):
                break
            entry = entries[v[0]]
            if entry[3] == b"":
                if (n := self._peek("I", names_off + 4 * i)) is None:
                    corrupted.add(entry[0])
                    continue
                name = self._peek_string(u32(self.rva_to_offset(n[0])))
                if name == b"" or len(name) > MAX_EXPORT_NAME_SIZE:
                    corrupted.add(entry[0])
                else:
                    entry[3] = name
            if entry[2] and entry[3] != b"" and b"." in entry[3]:
                if (n := self._peek("I", names_off + 4 * i)) is not None:
                    name = self._peek_string(u32(self.rva_to_offset(n[0])))
                    if name != b"" and len(name) <= MAX_EXPORT_NAME_SIZE and _is_printable(name):
                        entry[3] = name
        return [(_str(e[3]), e[1]) for e in entries if e[0] not in corrupted]

    @property
    def exported_functions(self):
        """ Names of the exported functions: the export entries with a name (LIEF's exported_functions). """
        return [n for n, _ in self.exports or [] if n not in ("", b"")]

    @property
    def has_delay_imports(self):
        return len(self.delay_imports) > 0

    @property
    def has_exports(self):
        return self.exports is not None

    @property
    def has_imports(self):
        return len(self.imports) > 0

    @property
    def has_tls(self):
        """ LIEF's Parser::parse_tls succeeds as soon as the TLS directory structure can be read. """
        def _parse():
            if (dd := self._dir(DIR_TLS)) is None or dd[0] == 0:
                return False
            return self._peek("4Q2I" if self.is_pe32plus else "6I", self.rva_to_offset(dd[0])) is not None
        return self._cached("tls", _parse)

    @property
    def imported_apis(self):
        return {(dll, api) for dll, entries in self.imports for api in entries}

    @property
    def imported_dlls(self):
        return {dll.lower() for dll, _ in self.imports}

    # ------------------------------------------------------------------------------------------------ sections
    @property
    def entrypoint(self):
        return self.rva_to_offset(self.addressof_entrypoint)

    @property
    def entrypoint_section(self):
        s = self.section_from_rva(self.addressof_entrypoint)
        return NullSection() if s is None else s

    @property
    def overlay(self):
        """ LIEF's Parser::parse_overlay: the data after the farthest end of the sections' raw data. """
        end = max([s.offset + s.size for s in self.sections] or [0])
        return self.data[end:] if end < len(self.data) else b""

    @property
    def real_section_names(self):
        """ Real names of the sections whose name is an offset in the string table ("/123"), as pbox computes them. """
        def _compute():
            r = {}
            for s in self.sections:
                n = s.name
                if not isinstance(n, str) or not (n.startswith("/") and n[1:].isdigit()):
                    continue
                o = self.pointerto_symbol_table + self.numberof_symbols * 18 + int(n[1:])
                end = self.data.find(b"\0", o)
                r[n] = self.data[o:end if end >= 0 else len(self.data)].decode("utf-8", errors="ignore")
            return r
        return self._cached("real_names", _compute)

    @property
    def empty_name_sections(self):
        return [s for s in self if s.real_name == ""]

    @property
    def known_packer_sections(self):
        return [s for s in self if s.real_name in self.common_packer_section_names]

    @property
    def non_standard_sections(self):
        return [s for s in self if s.real_name not in [""] + self.standard_section_names]

    @property
    def standard_sections(self):
        return [s for s in self if s.real_name in [""] + self.standard_section_names]

    # ------------------------------------------------------------------------------------------------ content
    @property
    def entropy(self):
        """ Entropy of the whole file. """
        return block_entropy(self.data)

    def average_block_entropy_per_section(self, blocksize=256, ignore_half_block_zeros=True, overlay=True, raw=True):
        r, t = 0., 0
        for e, w in [(block_entropy(s.bytes, blocksize, ignore_half_block_zeros),
                      len(s.bytes) if raw else s.size) for s in self]:
            if e == .0 or e[1] in [.0, None]:
                continue
            r += e[1] * w
            t += w
        if overlay:
            w = len(c := self.overlay)
            r += (block_entropy(c, blocksize, ignore_half_block_zeros)[1] or 0.) * w
            t += w
        return r / (t or 1)

    def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True,
                      ignore_overlay=False):
        key = ("block_entropy", blocksize, ignore_half_block_zeros, ignore_half_block_same_byte, ignore_overlay)
        def _compute():
            data = self.data[:len(self.data) - len(self.overlay)] if ignore_overlay else self.data
            return block_entropy(data, blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)
        return self._cached(key, _compute)

    def bytes_after_entrypoint(self, n):
        """ First n bytes of the section holding the entry point, padded with zeros (as pbox computes them). """
        content = self.entrypoint_section.content
        if content is None:
            raise PEError("the entry point is not in a section")
        return zeropad(n)(list(content[:n]))
