# -*- coding: UTF-8 -*-
import lief
import os
from bintropy import entropy
from capstone import *

from .elf import STD_SECTIONS as STD_SEC_ELF
from .pe import STD_SECTIONS as STD_SEC_PE


__all__ = ["block_entropy", "block_entropy_per_section", "disassemble_256B_after_ep", "entropy",
           "section_characteristics", "standard_sections"]


# selection of in-scope characteristics of Section objects for an executable parsed by LIEF
CHARACTERISTICS = ["characteristics", "entropy", "numberof_line_numbers", "numberof_relocations", "offset", "size",
                   "sizeof_raw_data", "virtual_size"]
STD_SECTIONS = {'ELF': STD_SEC_ELF, 'PE': STD_SEC_PE}

CS2CS_MODE = {
    CS_ARCH_ARM:   {32: CS_MODE_ARM, 64: CS_MODE_ARM},
    CS_ARCH_ARM64: {32: CS_MODE_ARM, 64: CS_MODE_ARM},
    CS_ARCH_MIPS:  {32: CS_MODE_MIPS32, 64: CS_MODE_MIPS64},
    CS_ARCH_PPC:   {32: CS_MODE_32, 64: CS_MODE_64},
    CS_ARCH_X86:   {32: CS_MODE_32, 64: CS_MODE_64},
}
LIEF2CS_ARCH = {
    lief.ELF.ARCH.ARM:    {32: CS_ARCH_ARM, 64: CS_ARCH_ARM64},
    lief.ELF.ARCH.MIPS:   {32: CS_ARCH_MIPS, 64: CS_ARCH_MIPS},
    lief.ELF.ARCH.PPC:    {32: CS_ARCH_PPC, 64: CS_ARCH_PPC},
    lief.ELF.ARCH.i386:   {32: CS_ARCH_X86, 64: CS_ARCH_X86},
    lief.ELF.ARCH.x86_64: {32: CS_ARCH_X86, 64: CS_ARCH_X86},
}

__cache = {}


def _parse_binary(f):
    def _wrapper(executable, *args, **kwargs):
        if str(executable) in __cache:
            binary = __cache[str(executable)]
        else:
            # try to parse the binary first ; capture the stderr messages from LIEF
            tmp_fd, null_fd = os.dup(2), os.open(os.devnull, os.O_RDWR)
            os.dup2(null_fd, 2)
            binary = lief.parse(str(executable))
            os.dup2(tmp_fd, 2)  # restore stderr
            os.close(null_fd)
            if binary is None:
                raise OSError("Unknown format")
            __cache[str(executable)] = binary
        return f(binary, *args, **kwargs)
    return _wrapper


@_parse_binary
def disassemble_Nbytes_after_ep(binary, n=256):
    addr = binary.header.entrypoint
    idc  = {lief.ELF.ELF_CLASS.CLASS32: 32, lief.ELF.ELF_CLASS.CLASS64: 64}[binary.header.identity_class]
    arch = LIEF2CS_ARCH[binary.header.machine_type][idc]
    mode = CS2CS_MODE[arch][idc]
    code = bytes(binary.get_content_from_virtual_address(addr, n))
    return [i.mnemonic for i in Cs(arch, mode).disasm(code, addr)]


# compute (name, data) pair for a Section object
_chars = lambda s: (s.name, {k: getattr(s, k, None) for k in CHARACTERISTICS})
# compute (name, entropy) pair for a Section object
_entr = lambda s, bs=0, z=False: (s.name, entropy(s.content, bs, z))

block_entropy             = lambda bsize: lambda exe: entropy(exe.read_bytes(), bsize, True)
block_entropy_per_section = lambda bsize: _parse_binary(lambda exe: [_entr(s, bsize, True) for s in exe.sections])
section_characteristics   = _parse_binary(lambda exe: {n: d for n, d in [_chars(s) for s in exe.sections]})
standard_sections         = _parse_binary(lambda exe: [s.name for s in exe.sections if s.name in \
                                                       STD_SECTIONS.get(exe.format.name, [])])

