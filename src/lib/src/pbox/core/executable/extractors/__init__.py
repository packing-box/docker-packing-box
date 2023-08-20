# -*- coding: UTF-8 -*-
from .__common__ import *
from .elf import *
from .macho import *
from .pe import *
from ....helpers.formats import expand_formats, FORMATS


__all__ = ["Extractors"]


EXTRACTORS = {
    'All': {
        '256B_block_entropy':             block_entropy(256),
        '256B_block_entropy_per_section': block_entropy_per_section(256),
        '512B_block_entropy':             block_entropy(512),
        '512B_block_entropy_per_section': block_entropy_per_section(512),
        'disassemble_256B_after_ep':      disassemble_Nbytes_after_ep,
        'entropy':                        lambda exe: entropy(exe.read_bytes()),
        'entrypoint':                     parse_binary(lambda exe: exe.abstract.entrypoint),
        'exported_functions':             parse_binary(lambda exe: exe.abstract.exported_functions),
        'imported_functions':             parse_binary(lambda exe: exe.abstract.imported_functions),
        'relocations':                    parse_binary(lambda exe: exe.abstract.relocations),
        'section_characteristics':        section_characteristics,
        'standard_sections':              standard_sections,
        'symbols':                        parse_binary(lambda exe: exe.abstract.symbols),
    },
    'ELF': {
        'elfeats': elfeats,
    },
    'Mach-O': {
        'mofeats': mofeats,
    },
    'MSDOS': {
        'pefeats': pefeats,
    },
    'PE': {
        'pefeats': pefeats,
    },
}


class Extractors(dict):
    """ This class represents the dictionary of features to be extracted for a given list of executable formats. """
    registry = None
    
    def __init__(self, exe):
        # parse YAML features definition once
        if Extractors.registry is None:
            Extractors.registry = {}
            # consider most specific features first, then intermediate classes and finally the collapsed class "All"
            for flist in [expand_formats("All"), list(FORMATS.keys())[1:], ["All"]]:
                for f in flist:
                    for name, func in EXTRACTORS.get(f, {}).items():
                        for f2 in expand_formats(f):
                            Extractors.registry.setdefault(f2, {})
                            Extractors.registry[f2].setdefault(name, func)
        if exe is not None:
            for name, func in Extractors.registry.get(exe.format, {}).items():
                self[name] = func(exe)

