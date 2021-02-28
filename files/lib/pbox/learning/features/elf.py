# -*- coding: UTF-8 -*-
from ast import literal_eval
from collections import OrderedDict
from elftools.elf.elffile import ELFFile


__all__ = ["elfeats", "ELFEATS"]


ELFEATS = {
    'magic': "Sum of magic bytes",
    # 7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00 
    # 7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
    # 7f 45 4c 46 02 01 01 03 00 00 00 00 00 00 00 00 
    #  NB: the sum of these numbers give distinct results, so no loss of information
    'data': "Data",
    'type': "Type",
    'machine': "Machine",
    'version': "Version",
    'entry_address': "Entry point address",
    'abi_version': "OS/ABI",
    'flags': "Flags",
    'size_of_elf_header': "Size of ELF header",
    'start_prg_headers': "Start of program headers",
    'size_prg_headers': "Size of program headers",
    'number_prg_headers': "Number of program headers",
    'start_sec_headers': "Start of section headers",
    'size_sec_headers': "Size of section headers",
    'number_sec_headers': "Number of section headers",
    'sec_header_str_table_idx': "Section header string table index",
}


def elfeats(executable):
    """ This uses features computed in ELF-Miner to extract +300 features from ELF files. """
    elf = ELFFile(executable)
    return {
        'magic': sum(elf.e_ident_raw),
        'data': elf.header['e_ident']['EI_DATA'],
        'type': elf.header['e_type'],
        'machine': elf.header['e_machine'],
        'version': elf.header['e_ident']['EI_VERSION'],
        'entry_address': elf.header['e_entry'],
        'os_abi': elf.header['e_ident']['EI_OSABI'],
        'abi_version': elf.header['e_ident']['EI_ABIVERSION'],
        'flags': elf.header['e_flags'],
        'size_of_elf_header': elf.header['e_ehsize'],
        'start_prg_headers': elf.header['e_phoff'],
        'size_prg_headers': elf.header['e_phentsize'],
        'number_prg_headers': elf.header['e_phnum'],
        'start_sec_headers': elf.header['e_shoff'],
        'size_sec_headers': elf.header['e_shentsize'],
        'number_sec_headers': elf.header['e_shnum'],
        'sec_header_str_table_idx': elf.header['e_shstrndx'],
    }

