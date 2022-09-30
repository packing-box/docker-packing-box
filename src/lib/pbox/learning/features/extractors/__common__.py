# -*- coding: UTF-8 -*-
import lief
import os
from bintropy import entropy


__all__ = ["block_entropy", "block_entropy_per_section", "entropy", "section_characteristics", "standard_sections"]


# selection of in-scope characteristics of Section objects for an executable parsed by LIEF
CHARACTERISTICS = ["characteristics", "entropy", "numberof_line_numbers", "numberof_relocations", "offset", "size",
                   "sizeof_raw_data", "virtual_size"]
STD_SECTIONS = {
    # https://www.cs.cmu.edu/afs/cs/academic/class/15213-f00/docs/elf.pdf
    'ELF': [".bss", ".comment", ".conflict", ".data", ".data1", ".debug", ".dynamic", ".dynstr", ".fini", ".got",
            ".gptab", ".hash", ".init", ".interp", ".liblist", ".line", ".lit4", ".lit8", ".note", ".plt", ".reginfo",
            ".rodata", ".rodata1", ".sbss", ".sdata", ".shstrtab", ".strtab", ".symtab", ".tdesc", ".text"],
    # https://github.com/roussieau/masterthesis/blob/master/src/detector/tools/pefeats/pefeats.cpp
    'PE': [".bss",".cormeta", ".data", ".debug", ".debug$F", ".debug$P", ".debug$S", ".debug$T", ".drective", ".edata",
           ".idata", ".idlsym", ".pdata", ".rdata", ".reloc", ".rsrc", ".sbss", ".sdata", ".srdata", ".sxdata", ".text",
           ".tls", ".tls$", ".vsdata", ".xdata"]
}
# possible relocation sections for the ELF format
for x in STD_SECTIONS['ELF'][:]:
    STD_SECTIONS['ELF'].append(".rel" + x)

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


# compute (name, data) pair for a Section object
_chars = lambda s: (s.name, {k: getattr(s, k, None) for k in CHARACTERISTICS})
# compute (name, entropy) pair for a Section object
_entr = lambda s, bs=0, z=False: (s.name, entropy(s.content, bs, z))

block_entropy             = lambda bsize: lambda exe: entropy(exe.read_bytes(), bsize, True)
block_entropy_per_section = lambda bsize: _parse_binary(lambda exe: [_entr(s, bsize, True) for s in exe.sections])
section_characteristics   = _parse_binary(lambda exe: [_chars(s) for s in exe.sections])
standard_sections         = _parse_binary(lambda exe: [s.name for s in exe.sections if s.name in \
                                                       STD_SECTIONS.get(exe.format.name, [])])

