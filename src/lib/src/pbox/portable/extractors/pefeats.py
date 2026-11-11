# -*- coding: UTF-8 -*-
""" Port of pefeats (https://github.com/packing-box/pefeats, src/pefeats.cpp) to Python, standard library only.

pefeats extracts 119 values from a PE file with PeLib (the version bundled with pefeats). This port computes the same
 values, as pefeats prints them, from the bytes of the file. It follows pefeats.cpp and the PeLib routines it relies on
 (PeHeader, ImportDirectory, IatDirectory, DebugDirectory, ResourceDirectory), including their behaviour on malformed
 files: integer sizes and wrap-arounds (unsigned int, dword, 64-bit), zero-filled buffers when reading beyond the end
 of the file, the state of the input stream between reads (see _Stream), float (32-bit) arithmetic of the entropies.

Values (in this order, as numbered in pefeats.cpp):
   1-8   DLL characteristics flags            42      raw data pointer not aligned
   9-21  header fields                        43-48   entropies (code, data, resources, header, file, EP section)
  22-41  sections                             49-112  64 bytes at the entry point
 113-116 imports                              117     IAT entries ; 118 no debug directory ; 119 resources

Unlike pefeats, it does not read memory out of bounds: where pefeats does (entry point in its valid range but in no
 section, see _ep_section), the values cannot be reproduced and PefeatsError is raised.
"""
import math
import struct
from collections import Counter


__all__ = ["pefeats", "pefeats_values", "PefeatsError", "FEATURES"]

FEATURES = [f"dll_characteristics_{i}" for i in range(1, 9)] + [
    "checksum", "image_base", "base_of_code", "os_major_version", "os_minor_version", "size_of_image", "size_of_code",
    "size_of_headers", "size_of_initialized_data", "size_of_uninitialized_data", "size_of_stack_reserve",
    "size_of_stack_commit", "section_alignment", "number_standard_sections", "number_non_standard_sections",
    "ratio_standard_sections", "number_x_sections", "number_w_sections", "number_wx_sections", "number_rx_sections",
    "number_rw_sections", "number_rwx_sections", "code_section_not_x", "x_section_not_code", "no_code_section",
    "ep_not_in_code_section", "ep_not_in_standard_section", "ep_not_in_x_section", "ep_raw_virtual_size_ratio",
    "number_sections_zero_raw_size", "number_sections_virtual_size_bigger", "max_raw_virtual_size_ratio",
    "min_raw_virtual_size_ratio", "raw_data_pointer_not_aligned", "entropy_code", "entropy_data", "entropy_resources",
    "entropy_header", "entropy_file", "entropy_ep_section"] + [f"byte_{i}_after_ep" for i in range(64)] + [
    "number_imported_dlls", "number_imported_functions", "number_malicious_apis", "ratio_malicious_apis",
    "number_iat_entries", "no_debug_directory", "number_resources"]

STANDARD_SECTION_NAMES = [b".bss", b".cormeta", b".data", b".debug$F", b".debug$P", b".debug$S", b".debug$T",
                          b".drective", b".edata", b".idata", b".idlsym", b".pdata", b".rdata", b".reloc", b".rsrc",
                          b".sbss", b".sdata", b".srdata", b".sxdata", b".text", b".tls", b".tls$", b".vsdata",
                          b".xdata"]
MALICIOUS_APIS = [b"GetProcAddress", b"LoadLibraryA", b"LoadLibrary", b"ExitProcess", b"GetModuleHandleA",
                  b"VirtualAlloc", b"VirtualFree", b"GetModuleFileNameA", b"CreateFileA", b"RegQueryValueExA",
                  b"MessageBoxA", b"GetCommandLineA", b"VirtualProtect", b"GetStartupInfoA", b"GetStdHandle",
                  b"RegOpenKeyExA"]

# PeLib's limits (PeLibAux.h)
MAX_IMPORT_DLLS, MAX_IMPORTED_FUNCTIONS, IMPORT_NAME_MAX_LENGTH = 0x100, 0x1000, 96

U16, U32, U64 = 0xffff, 0xffffffff, 0xffffffffffffffff


class PefeatsError(ValueError):
    """ pefeats would fail (no output) or its output cannot be reproduced. """


def _f32(x):
    """ Round a float to single precision (C's float). """
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _entropy(data, size=None):
    """ pefeats' _ShannonEntropyV on 'size' bytes: data followed by zeros up to size (as a zero-initialized buffer
         partially filled from the file). Computed in float as in C: log2f of float arguments, float accumulator. """
    size = len(data) if size is None else size
    if size == 0:
        return 0.
    cnt = Counter(data[:size])
    count = [cnt.get(b, 0) for b in range(256)]
    count[0] += size - min(len(data), size)
    e = 0.
    for c in count:
        if c > 0:
            e = _f32(e + (-1. * c) / size * _f32(math.log2(_f32(c / size))))
    return e


def _fmt(x):
    """ std::fixed with setprecision(3), for a float value. """
    return "%.3f" % x


# ---------------------------------------------------------------------------------------------------- input stream
class _Stream:
    """ Model of the std::ifstream that PeLib reads from: the position and the fail/eof flags, as they evolve with
         seekg, read and tellg (libstdc++ semantics), so that read failures propagate as in PeLib. """
    def __init__(self, data):
        self.data, self.size, self.pos, self.fail, self.eof = data, len(data), 0, False, False

    def __bool__(self):
        return not self.fail

    def clear(self):
        self.fail = self.eof = False

    def seekg(self, pos):
        self.eof = False
        if self.fail:
            return
        if pos < 0 or pos > 0x7fffffffffffffff:   # streamoff is signed (64 bits)
            self.fail = True
            return
        self.pos = pos   # seeking beyond the end of a file succeeds

    def tellg(self):
        return -1 if self.fail else self.pos

    def read(self, n):
        """ Bytes read (possibly fewer than n, then the stream fails). """
        if self.fail:
            return b""
        chunk = self.data[self.pos:self.pos+n] if self.pos < self.size else b""
        self.pos += len(chunk)
        if len(chunk) < n:
            self.fail = self.eof = True
        return chunk

    def file_size(self):
        """ PeLib's fileSize(stream): -1 (as an unsigned 64-bit value) when the stream has failed. """
        return U64 if self.fail else self.size

    def wrapper(self):
        """ PeLib's IStreamWrapper: saves the position and the state, then clears the stream ; restore() puts back the
             saved state (or-ed with the current one) and the position (which fails silently if the stream failed). """
        saved = (self.pos if not self.fail else -1, self.fail, self.eof)
        self.clear()
        def restore():
            self.fail, self.eof = self.fail or saved[1], self.eof or saved[2]
            self.seekg(saved[0])
        return restore

    def get_string(self, offset, max_length=0):
        """ PeLib's getStringFromFileOffset. """
        restore = self.wrapper()
        result = b""
        self.clear()
        self.seekg(offset)
        if self:
            while True:
                c = self.read(1)
                if not self or c == b"\0":
                    break
                result += c
                if max_length and len(result) == max_length:
                    break
        restore()
        return result


def _u(fmt, data, offset):
    return struct.unpack_from("<" + fmt, data, offset)


# --------------------------------------------------------------------------------------------------------- PE header
class _PeHeader:
    """ PeLib's PeHeaderT<bits>, as read by MzHeader::read and PeHeaderT<bits>::read. """
    def __init__(self, data):
        self.data, self.file_size = data, len(data)
        if len(data) < 64 or data[:2] != b"MZ":
            raise PefeatsError("not a PE file")   # openPeFile fails: "Invalid PE File"
        self.e_lfanew = nt = _u("I", data, 0x3c)[0]
        # file type: 64-bit only for AMD64/IA64 machines with the PE32+ magic (getFileType), otherwise 32-bit
        hdr32 = data[nt:nt+4+20+96].ljust(4 + 20 + 96, b"\0") if nt < len(data) else b"\0" * 120
        machine, magic = _u("H", hdr32, 4)[0], _u("H", hdr32, 24)[0]
        self.bits = 64 if machine in (0x8664, 0x200) and magic == 0x20b else 32
        self.opt_size = 112 if self.bits == 64 else 96
        self.var_max = U64 if self.bits == 64 else U32
        size = 4 + 20 + self.opt_size
        h = data[nt:nt+size].ljust(size, b"\0") if nt < len(data) else b"\0" * size
        (self.number_of_sections, ) = _u("H", h, 6)
        (self.pointer_to_symbol_table, self.number_of_symbols, self.size_of_optional_header) = _u("IIH", h, 12)
        o = 24
        (self.size_of_code, self.size_of_initialized_data, self.size_of_uninitialized_data,
         self.address_of_entry_point, self.base_of_code) = _u("5I", h, o + 4)
        if self.bits == 64:
            self.image_base, = _u("Q", h, o + 24)
        else:
            self.image_base, = _u("I", h, o + 28)
        (self.section_alignment, self.file_alignment, self.major_os_version, self.minor_os_version) = \
            _u("IIHH", h, o + 32)
        (self.size_of_image, self.size_of_headers, self.checksum, _, self.dll_characteristics) = \
            _u("IIIHH", h, o + 56)
        if self.bits == 64:
            self.size_of_stack_reserve, self.size_of_stack_commit = _u("QQ", h, o + 72)
            self.number_of_rva_and_sizes, = _u("I", h, o + 108)
        else:
            self.size_of_stack_reserve, self.size_of_stack_commit = _u("II", h, o + 72)
            self.number_of_rva_and_sizes, = _u("I", h, o + 92)
        self._read_data_directories(nt + size)
        self._read_sections((nt + 4 + 20 + self.size_of_optional_header) & U32)

    def _read_data_directories(self, offset):
        """ PeHeaderT::readDataDirectories: at most 16, stopping at the end of the file ; a directory cut by the end
             of the file gets the bytes of the previous one where data is missing (the buffer is reused). """
        self.dirs, buf = [], bytearray(8)
        for _ in range(min(self.number_of_rva_and_sizes, 16)):
            if offset >= self.file_size:
                break
            if offset + 8 > self.file_size and offset + 4 > self.file_size:
                break
            chunk = self.data[offset:offset+8]
            buf[:len(chunk)] = chunk
            self.dirs.append(_u("II", bytes(buf), 0))
            offset += 8

    def _read_sections(self, offset):
        """ PeHeaderT::readSections: the section headers that fit in the file. """
        self.sections = []
        for _ in range(self.number_of_sections):
            if (offset + 40) & U32 > self.file_size:
                break
            name = self.data[offset:offset+8]
            vsize, va, raw_size, raw_ptr = _u("IIII", self.data, offset + 8)
            characteristics, = _u("I", self.data, offset + 36)
            self.sections.append((name.replace(b"\0", b""), vsize, va, raw_size, raw_ptr, characteristics))
            offset = (offset + 40) & U32

    # accessors on sections: (name, virtual size, virtual address, raw size, raw pointer, characteristics)
    def name(self, i):
        return self.sections[i][0]

    def vsize(self, i):
        return self.sections[i][1]

    def va(self, i):
        return self.sections[i][2]

    def raw_size(self, i):
        return self.sections[i][3]

    def raw_ptr(self, i):
        return self.sections[i][4]

    def characteristics(self, i):
        return self.sections[i][5]

    def dir_rva(self, i):
        return self.dirs[i][0] if i < len(self.dirs) else 0

    def dir_size(self, i):
        return self.dirs[i][1] if i < len(self.dirs) else 0

    def size(self):
        """ PeHeaderT::size: NT headers (with the directories read) and the declared number of sections. """
        return 4 + 20 + self.opt_size + 8 * len(self.dirs) + self.number_of_sections * 40

    def section_with_rva(self, rva):
        """ PeHeaderT::getSectionWithRva: 0xFFFF if none. """
        act, detected = 0, False
        for i in range(len(self.sections)):
            m = self.vsize(i) if self.vsize(i) >= self.raw_size(i) else self.raw_size(i)
            if self.va(i) <= rva and (self.va(i) + m) & U32 > rva:
                act_max = self.vsize(act) if self.vsize(act) >= self.raw_size(act) else self.raw_size(act)
                if not detected or self.va(i) > self.va(act) or (self.va(i) == self.va(act) and m < act_max):
                    act, detected = i, True
        return act if detected else U16

    def rva_to_offset(self, rva):
        """ PeHeaderT::rvaToOffset (VAR4_8: dword for 32 bits, qword for 64 bits). """
        mask = self.var_max
        if rva < 0x1000:
            return rva
        s = self.section_with_rva(rva)
        if s == U16 or rva > (self.va(s) + self.raw_size(s)) & U32:
            return mask
        if self.raw_ptr(s) < self.file_alignment:
            return (rva - self.va(s)) & mask
        return (self.raw_ptr(s) + (rva - self.va(s))) & mask

    def is_valid_rva(self, rva):
        """ PeHeaderT::isValidRva. """
        if rva >= self.size_of_image:
            return False
        if len(self.sections) == 0 or rva < self.file_alignment:
            return True
        a = self.section_alignment
        for i in range(len(self.sections)):
            size = self.vsize(i) or self.raw_size(i)
            size = (size + (a - 1)) & ~(a - 1) & U32 if a else 0   # AlignToSize, in uint32
            if self.va(i) <= rva < (self.va(i) + size) & U32:
                return True
        return False

    def calc_start_of_code(self):
        """ PeHeaderT::calcStartOfCode. """
        m = 0xffffffff
        for i in range(min(len(self.dirs), 15)):
            if (rva := self.dir_rva(i)) and (o := self.rva_to_offset(rva)) < m:
                m = o
        for i in range(len(self.sections)):
            if (self.raw_ptr(i) < m or m == 0xffffffff) and self.raw_size(i):
                if self.raw_ptr(i):
                    m = self.raw_ptr(i)
        return m & U32

    def calc_size_of_image(self):
        """ PeHeaderT::calcSizeOfImage: from the (first) section with the highest virtual address. """
        last = max(range(len(self.sections)), key=lambda i: (self.va(i), -i))
        if self.vsize(last):
            return (self.va(last) + self.vsize(last)) & U32
        return (self.va(last) + max(self.vsize(last), self.raw_size(last))) & U32

    def ep_is_valid(self):
        ep = self.address_of_entry_point
        return ep >= self.calc_start_of_code() and ep <= self.size_of_image


# --------------------------------------------------------------------------------------------------------- features
def _header(h):
    c = h.dll_characteristics
    flags = [int(bool(c & f)) for f in (0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x2000, 0x8000)]
    return [str(x) for x in flags + [h.checksum, h.image_base, h.base_of_code, h.major_os_version,
            h.minor_os_version, h.size_of_image, h.size_of_code, h.size_of_headers, h.size_of_initialized_data,
            h.size_of_uninitialized_data, h.size_of_stack_reserve, h.size_of_stack_commit, h.section_alignment]]


def _ep_section(h):
    s = h.section_with_rva(h.address_of_entry_point)
    if s == U16:
        # pefeats then reads the section headers out of bounds (undefined behaviour)
        raise PefeatsError("the entry point is in its valid range but in no section")
    return s


def _sections(h):
    n = len(h.sections)
    if n == 0:
        raise PefeatsError("no section")   # pefeats exits with code 3
    std = wr = ex = wx = rx = rw = rwx = zero_raw = vbigger = 0
    max_ratio, min_ratio = -1., 1e6
    code_not_x = x_not_code = ptr_not_aligned = False
    no_code = True
    for i in range(n):
        name, ch, srd, vs = h.name(i), h.characteristics(i), h.raw_size(i), h.vsize(i)
        std += name in STANDARD_SECTION_NAMES
        w, r, x = bool(ch & 0x80000000), bool(ch & 0x40000000), bool(ch & 0x20000000)
        wr += w
        if x:
            ex += 1
            if name != b".text" and name != b"tls":
                x_not_code = True
        rwx += w and r and x
        wx += w and x
        rx += r and x
        rw += r and w
        zero_raw += srd == 0
        if h.file_alignment == 0:
            raise PefeatsError("null file alignment")   # pefeats: division by zero (SIGFPE)
        if h.raw_ptr(i) % h.file_alignment != 0:
            ptr_not_aligned = True
        if vs != 0:
            ratio = (1. * srd) / vs
            if ratio > max_ratio:
                max_ratio = _f32(ratio)
            if ratio < min_ratio:
                min_ratio = _f32(ratio)
            if ratio < 1:
                vbigger += 1
        if name == b".text":
            no_code = False
            if not x:
                code_not_x = True
    if min_ratio == 1e6:
        min_ratio = -1.
    ratio_std = _f32((1. * std) / n)
    ep_ratio = -1.
    if h.ep_is_valid():
        s = _ep_section(h)
        ep_name, ep_ch = h.name(s), h.characteristics(s)
        no_ep_code, no_ep_std, no_ep_x = ep_name != b".text", ep_name not in STANDARD_SECTION_NAMES, \
                                         not ep_ch & 0x20000000
        if h.vsize(s) != 0:
            ep_ratio = _f32((1. * h.raw_size(s)) / h.vsize(s))
    else:
        no_ep_code = no_ep_std = no_ep_x = True
    b = lambda v: str(int(bool(v)))
    return [str(std), str(n - std), _fmt(ratio_std), str(ex), str(wr), str(wx), str(rx), str(rw), str(rwx),
            b(code_not_x), b(x_not_code), b(no_code), b(no_ep_code), b(no_ep_std), b(no_ep_x), _fmt(ep_ratio),
            str(zero_raw), str(vbigger), _fmt(max_ratio), _fmt(min_ratio), b(ptr_not_aligned)]


def _read_at(data, offset, size):
    """ ifstream seekg(offset) then read(size) into a zero-filled buffer of size bytes. """
    return data[offset:offset+size] if 0 <= offset < len(data) else b""


def _entropies(h):
    data = h.data
    e_ep = e_data = e_code = e_rsrc = -1.
    if h.ep_is_valid():
        s = _ep_section(h)
        e_ep = _entropy(_read_at(data, h.raw_ptr(s), h.raw_size(s)), h.raw_size(s))
    ids, i = {b".data": -1, b".text": -1, b".rsrc": -1}, 0
    while i < len(h.sections) and -1 in ids.values():
        if h.name(i) in ids:
            ids[h.name(i)] = i
        i += 1
    def _sec(i):
        return _entropy(_read_at(data, h.raw_ptr(i), h.raw_size(i)), h.raw_size(i))
    if ids[b".data"] != -1:
        e_data = _sec(ids[b".data"])
    if ids[b".text"] != -1:
        e_code = _sec(ids[b".text"])
    if ids[b".rsrc"] != -1:
        e_rsrc = _sec(ids[b".rsrc"])
    hs = h.size()
    e_header = _entropy(data[:hs], hs)
    fs = h.calc_size_of_image()
    e_file = _entropy(data[:fs], fs)
    return [_fmt(e) for e in (e_code, e_data, e_rsrc, e_header, e_file, e_ep)]


def _ep64(h):
    if not h.ep_is_valid():
        return ["-1"] * 64
    o = h.rva_to_offset(h.address_of_entry_point)
    # seekg takes a signed 64-bit offset: the maximum 64-bit value (-1) fails, hence nothing is read
    chunk = b"" if o > 0x7fffffffffffffff else _read_at(h.data, o, 64)
    return [str(x) for x in chunk.ljust(64, b"\0")]


# ------------------------------------------------------------------------------------------------------- directories
def _imports(h):
    """ ImportDirectory<bits>::read then the counts of pefeats' dumpImportDirectory ; None if the read fails. """
    if not (len(h.dirs) >= 2 and h.dir_rva(1)):
        return None
    st = _Stream(h.data)
    st.wrapper()
    bits, ptr = h.bits, ("Q" if h.bits == 64 else "I")
    psize = 8 if bits == 64 else 4
    ordinal_mask = 1 << (bits - 1)
    soi = h.size_of_image
    if h.dir_rva(1) > soi:
        return None
    fsize = st.file_size()
    rva = h.dir_rva(1)
    offset = h.rva_to_offset(rva) & U32
    if (offset + 20) & U32 > fsize:
        return None
    st.seekg(offset)
    descs, dlls, desc_offset = [], set(), offset
    while True:
        buf = b"\0" * 20
        if (desc_offset + 20) & U32 <= fsize:
            buf = st.read(20).ljust(20, b"\0")
        elif not h.is_valid_rva((rva + 20) & U32):
            break
        oft, _, _, name_rva, ft = _u("5I", buf, 0)
        desc_offset, rva = (desc_offset + 20) & U32, (rva + 20) & U32
        if name_rva == 0 or ft == 0:
            break
        if name_rva > soi or not h.is_valid_rva(name_rva):
            break
        if ft > soi:
            break
        name = st.get_string(h.rva_to_offset(name_rva), IMPORT_NAME_MAX_LENGTH)
        if name not in dlls:
            dlls.add(name)
            if len(dlls) > MAX_IMPORT_DLLS:
                break
        descs.append({'oft': oft, 'ft': ft, 'oft_list': [], 'ft_list': []})

    def valid_oft(d):   # hasValidOriginalFirstThunk
        return d['oft'] >= h.file_alignment and h.rva_to_offset(d['oft']) != h.var_max

    # OriginalFirstThunk (ILT)
    for d in descs:
        if not valid_oft(d):
            continue
        va = d['oft']
        value = 0
        st.clear()
        st.seekg(h.rva_to_offset(va) & U32)
        idx = 0
        while True:
            if fsize < (h.rva_to_offset(va) + psize) & U64:
                return None
            va = (va + psize) & U32
            chunk = st.read(psize)
            if len(chunk) == psize:
                value = _u(ptr, chunk, 0)[0]
            elif chunk:   # partial read: only the first bytes are overwritten
                value = _u(ptr, chunk + struct.pack("<" + ptr, value)[len(chunk):], 0)[0]
            if value == 0:
                break
            if idx >= MAX_IMPORTED_FUNCTIONS:
                break
            if value & ordinal_mask == 0 and value >= soi:
                break
            d['oft_list'].append(value)
            idx += 1
    # FirstThunk (IAT)
    for d in descs:
        has_ilt = valid_oft(d)
        va = d['ft']
        if not h.is_valid_rva(va):
            return None
        st.clear()
        st.seekg(h.rva_to_offset(va) & U32)
        idx = 0
        while True:
            if fsize < (h.rva_to_offset(va) + psize) & U64:
                return None
            va = (va + psize) & U32
            chunk = st.read(psize)
            value = _u(ptr, chunk.ljust(psize, b"\0"), 0)[0]
            if value == 0:
                break
            if idx >= MAX_IMPORTED_FUNCTIONS:
                break
            d['ft_list'].append(value)
            idx += 1
            if has_ilt and len(d['oft_list']) == len(d['ft_list']):
                break
    # names
    for d in descs:
        lst = d['oft_list'] if valid_oft(d) else d['ft_list']
        names = []
        for value in lst:
            if value & ordinal_mask:
                names.append(b"")
                continue
            st.seekg(h.rva_to_offset(value) & U32)
            st.read(2)
            if not st:
                return None
            names.append(st.get_string(st.tellg(), IMPORT_NAME_MAX_LENGTH))
        d['oft_names' if valid_oft(d) else 'ft_names'] = names
    # counts of pefeats
    n_funcs = sum(len(d['ft_list']) for d in descs)
    n_mal = 0
    for d in descs:
        for j in range(len(d['ft_list'])):
            # ImportDirectory::getFunctionName(OLDDIR)
            if d['oft'] and j < len(d['oft_list']):
                name = d.get('oft_names', [b""] * len(d['oft_list']))[j]
            else:
                name = d.get('ft_names', [b""] * len(d['ft_list']))[j]
            n_mal += name in MALICIOUS_APIS
    ratio = 0. if n_funcs == 0 else _f32((1. * n_mal) / n_funcs)
    return [str(len(descs)), str(n_funcs), str(n_mal), _fmt(ratio)]


def _iat(h):
    """ IatDirectoryT<bits>::read then calcNumberOfAddresses ; None if the read fails. """
    if not (len(h.dirs) >= 13 and h.dir_rva(12) and h.dir_size(12)):
        return None
    fsize = len(h.data)
    offset = h.rva_to_offset(h.dir_rva(12))
    if fsize <= offset:
        return None
    size = min(fsize - offset, h.dir_size(12))
    buf = h.data[offset:offset+size]
    n, cur, fs32 = 0, offset & U32, fsize & U32
    while cur < fs32:
        if (v := _u("I", buf[4*n:4*n+4].ljust(4, b"\0"), 0)[0]) == 0:
            break
        n += 1
        cur = (cur + 4) & U32
    return n


def _debug(h):
    """ DebugDirectoryT<bits>::read: True if it succeeds. """
    if not (len(h.dirs) >= 7 and h.dir_rva(6) and h.dir_size(6)):
        return False
    st = _Stream(h.data)
    st.wrapper()
    fsize = st.file_size()
    offset, size = h.rva_to_offset(h.dir_rva(6)) & U32, h.dir_size(6)
    if fsize < (offset + size) & U32:
        return False
    st.seekg(offset)
    buf = st.read(size).ljust(size, b"\0")
    for i in range(size // 28):
        size_of_data, _, ptr = _u("III", buf, 28 * i + 16)
        if ptr >= fsize or (ptr + size_of_data) & U32 >= fsize:
            return False
        st.seekg(ptr)
        st.read(size_of_data)
        if not st:
            return False
    return True


def _resources(h):
    """ ResourceDirectoryT<bits>::read then the number of occupied address ranges ; None if the read fails. """
    if not (len(h.dirs) >= 3 and h.dir_rva(2)):
        return None
    rsrc_rva = h.dir_rva(2)
    rsrc_off = h.rva_to_offset(rsrc_rva) & U32
    if not rsrc_off:
        return None
    st = _Stream(h.data)
    st.wrapper()
    fsize = st.file_size()
    if fsize < rsrc_off:
        return None
    st.seekg(rsrc_off)
    fs32, soi = fsize & U32, h.size_of_image
    ranges, node_offsets = [0], set()

    def leaf(off):
        if (rsrc_off + off + 16) & U32 > st.file_size():
            return False
        st.seekg((rsrc_off + off) & U32)
        to_data, size = _u("II", st.read(16).ljust(16, b"\0"), 0)
        ranges[0] += 1
        entry_size = min(size, fs32)
        if not (to_data - rsrc_rva + size) & U32:
            return True
        if (rsrc_off + ((to_data - rsrc_rva) & U32) + entry_size) & U32 > fs32:
            return True
        if to_data < rsrc_rva:
            return True
        st.seekg((rsrc_off + ((to_data - rsrc_rva) & U32)) & U32)
        st.read(entry_size)
        if entry_size > 0:
            ranges[0] += 1
        return True

    def node(off):
        if (rsrc_off + off + 16) & U32 > st.file_size():
            return False
        st.seekg((rsrc_off + off) & U32)
        hdr = st.read(16).ljust(16, b"\0")
        n = sum(_u("HH", hdr, 12))
        ranges[0] += 1
        if (rsrc_rva + n * 8) & U32 > soi:
            return True
        if (rsrc_off + off + 16 + n * 8) & U32 > st.file_size():
            return False
        children = st.read(n * 8).ljust(n * 8, b"\0")
        node_offsets.add(off)
        if n > 0:
            ranges[0] += 1
        for i in range(n):
            name, to_data = _u("II", children, 8 * i)
            last = st.tellg() & U32
            if name & 0x80000000:
                name_off = name & 0x7fffffff
                if name_off + 2 < st.file_size():
                    if ((rsrc_off + name_off) & U32) + 2 > st.file_size():
                        return False
                    st.seekg((rsrc_off + name_off) & U32)
                    length = _u("H", st.read(2).ljust(2, b"\0"), 0)[0]
                    if (rsrc_off + name_off + 2 * length) & U32 > st.file_size():
                        return False
                    st.read(2 * length)
                st.seekg(last)
            value = to_data & 0x7fffffff if to_data & 0x80000000 else to_data
            if value in node_offsets:
                return True
            if not (node(value) if to_data & 0x80000000 else leaf(value)):
                return False
            st.seekg(last)
        return True

    try:
        return ranges[0] if node(0) else None
    except RecursionError:
        raise PefeatsError("resource tree too deep") from None


# ------------------------------------------------------------------------------------------------------------- main
def pefeats(data):
    """ The 119 values that pefeats prints for the PE file whose bytes are given, as strings (in pefeats' format).

    Raises PefeatsError when pefeats would fail (no output) or when its output cannot be reproduced. """
    if not isinstance(data, (bytes, bytearray)):
        with open(data, "rb") as f:
            data = f.read()
    h = _PeHeader(bytes(data))
    values = _header(h) + _sections(h) + _entropies(h) + _ep64(h)
    imp = _imports(h)
    # NB: on failure, pefeats prints the ratio without std::fixed (another output stream), hence "-1"
    values += ["-1"] * 4 if imp is None else imp
    iat = _iat(h)
    values.append("-1" if iat is None else str(iat))
    values.append("0" if _debug(h) else "1")
    res = _resources(h)
    values.append("-1" if res is None else str(res))
    return values


def pefeats_values(data):
    """ The 119 values as pbox gets them from pefeats' output (integers and floats). """
    from ast import literal_eval
    return [literal_eval(v) for v in pefeats(data)]
