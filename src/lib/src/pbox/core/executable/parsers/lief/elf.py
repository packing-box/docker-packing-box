# -*- coding: UTF-8 -*-
from .__common__ import *


__all__ = ["ELF"]


def __init_elf():
    ELFSection = get_section_class("ELFSection")
    
    class ELF(Binary):
        checksum = None
        
        def __iter__(self):
            for s in self.sections:
                yield ELFSection(s)
        
        def _get_builder(self):
            return lief.ELF.Builder(self._parsed)
        
        @property
        def entrypoint(self):
            return self.virtual_address_to_offset(self._parsed.entrypoint)
        
        @property
        def entrypoint_section(self):
            return self.section_from_offset(self._parsed.entrypoint)
    
    ELF.__name__ = "ELF"
    return ELF
lazy_load_object("ELF", __init_elf)

