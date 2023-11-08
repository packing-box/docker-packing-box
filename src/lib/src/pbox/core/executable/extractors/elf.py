# -*- coding: UTF-8 -*-
#TODO: move this part to pyelftools parser (in pbox.core.executable.parsers.pyelftools) OR
#       create a C program to mimic the functionality of pefeats for ELF
from elftools.elf.elffile import ELFFile


__all__ = ["elfeats"]


_ELFEATS = {
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
    with executable.open('rb') as fh:
        elf = ELFFile(fh)
        headers = elf.header
    return {
        'magic': sum(elf.e_ident_raw),
        'data': headers['e_ident']['EI_DATA'],
        'type': headers['e_type'],
        'machine': headers['e_machine'],
        'version': headers['e_ident']['EI_VERSION'],
        'entry_address': headers['e_entry'],
        'os_abi': headers['e_ident']['EI_OSABI'],
        'abi_version': headers['e_ident']['EI_ABIVERSION'],
        'flags': headers['e_flags'],
        'size_of_elf_header': headers['e_ehsize'],
        'start_prg_headers': headers['e_phoff'],
        'size_prg_headers': headers['e_phentsize'],
        'number_prg_headers': headers['e_phnum'],
        'start_sec_headers': headers['e_shoff'],
        'size_sec_headers': headers['e_shentsize'],
        'number_sec_headers': headers['e_shnum'],
        'sec_header_str_table_idx': headers['e_shstrndx'],
    }

