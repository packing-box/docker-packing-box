# -*- coding: UTF-8 -*-
from bintropy import entropy
from capstone import *
from lief import parse, ARCHITECTURES, ELF, MachO, PE

from ....helpers.data import get_data


__all__ = ["block_entropy", "block_entropy_per_section", "disassemble_Nbytes_after_ep", "entropy", "parse_binary",
           "section_characteristics", "standard_sections"]


# selection of in-scope characteristics of Section objects for an executable parsed by LIEF
CHARACTERISTICS = ["characteristics", "entropy", "flags", "numberof_line_numbers", "numberof_relocations", "offset",
                   "original_size", "size", "sizeof_raw_data", "type", "virtual_size"]
CS2CS_MODE = {
    CS_ARCH_ARM:   {32: CS_MODE_ARM,    64: CS_MODE_ARM},
    CS_ARCH_ARM64: {32: CS_MODE_ARM,    64: CS_MODE_ARM},
    CS_ARCH_MIPS:  {32: CS_MODE_MIPS32, 64: CS_MODE_MIPS64},
    CS_ARCH_PPC:   {32: CS_MODE_32,     64: CS_MODE_64},
    CS_ARCH_X86:   {32: CS_MODE_32,     64: CS_MODE_64},
}
LIEF2CS_ARCH = {
    ARCHITECTURES.ARM:   {32: CS_ARCH_ARM,   64: CS_ARCH_ARM64},
    ARCHITECTURES.ARM64: {32: CS_ARCH_ARM,   64: CS_ARCH_ARM64},
    ARCHITECTURES.INTEL: {32: CS_ARCH_X86,   64: CS_ARCH_X86},
    ARCHITECTURES.MIPS:  {32: CS_ARCH_MIPS,  64: CS_ARCH_MIPS},
    ARCHITECTURES.PPC:   {32: CS_ARCH_PPC,   64: CS_ARCH_PPC},
    ARCHITECTURES.SPARC: {32: CS_ARCH_SPARC, 64: CS_ARCH_SPARC},
    ARCHITECTURES.SYSZ:  {32: CS_ARCH_SYSZ,  64: CS_ARCH_SYSZ},
    ARCHITECTURES.X86:   {32: CS_ARCH_X86,   64: CS_ARCH_X86},
    ARCHITECTURES.XCORE: {32: CS_ARCH_XCORE, 64: CS_ARCH_XCORE},
}


def parse_binary(f):
    def _wrapper(target, *args, **kwargs):
        return f(parse(str(target)), *args, **kwargs)
    return _wrapper


@parse_binary
def disassemble_Nbytes_after_ep(binary, n=256):
    ep   = binary.abstract.header.entrypoint
    idc  = [32, 64][binary.abstract.header.is_64]
    arch = LIEF2CS_ARCH[binary.abstract.header.architecture][idc]
    mode = CS2CS_MODE.get(arch, {}).get(idc, CS_MODE_LITTLE_ENDIAN)
    return [i.mnemonic for i in Cs(arch, mode).disasm(bytes(binary.get_content_from_virtual_address(ep, n)), ep)]


# compute (name, data) pair for a Section object
_chars = lambda s: (s.name, {k: getattr(s, k, None) for k in CHARACTERISTICS})
# compute (name, entropy) pair for a Section object
_entr = lambda s, bs=0, z=False: (s.name, entropy(s.content, bs, z))

block_entropy             = lambda bsize: lambda exe: entropy(exe.read_bytes(), bsize, True)
block_entropy_per_section = lambda bsize: parse_binary(lambda exe: [_entr(s, bsize, True) for s in exe.sections])
section_characteristics   = parse_binary(lambda exe: {n: d for n, d in [_chars(s) for s in exe.sections]})
standard_sections         = parse_binary(lambda exe: [s.name for s in exe.sections if s.name in \
                                                      get_data(exe.format.name)['COMMON_SECTION_NAMES']])

