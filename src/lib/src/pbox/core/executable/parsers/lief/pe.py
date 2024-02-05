# -*- coding: UTF-8 -*-
from .__common__ import *


__all__ = ["PE"]


def __init_pe():
    _p = cached_property
    sec_chars = {k: v for k, _, v in getattr(lief.PE.Section.CHARACTERISTICS, "@entries").values()}
    sec_types = {k: v for k, _, v in getattr(lief.PE.SECTION_TYPES, "@entries").values()}
    PESection = get_section_class("PESection",
        characteristics="characteristics",
        has_slack_space=_p(lambda s: s.size > s.virtual_size),
        is_executable=_p(lambda s: s.characteristics & s.CHARACTERISTICS['MEM_EXECUTE'].value > 0),
        is_readable=_p(lambda s: s.characteristics & s.CHARACTERISTICS['MEM_READ'].value > 0),
        is_writable=_p(lambda s: s.characteristics & s.CHARACTERISTICS['MEM_WRITE'].value > 0),
        raw_data_size=_p(lambda s: s.size),
        real_name="name",
        virtual_size="virtual_size",
        CHARACTERISTICS=sec_chars,
        TYPES=sec_types,
    )
    
    class PE(Binary):
        _build_config = BuildConfig("dos_stub", "imports", "overlay", "relocations", "resources", "tls",
                                    "patch_imports")
        
        def __iter__(self):
            for s in self.sections:
                s = PESection(s, self)
                if hasattr(self, "_real_section_names"):
                    s.real_name = self.real_section_names.get(s.name, s.name)
                yield s
        
        def _get_builder(self):
            builder = lief.PE.Builder(self._parsed)
            for instruction, flag in list(self._build_config.items()):
                getattr(builder, instruction if instruction.startswith("patch") else f"build_{instruction}")(flag)
                del self._build_config[instruction]
            builder.build_overlay(False)  # build_overlay(True) fails when adding a section to the binary
            return builder
        
        @property
        def api_imports(self):
            return sorted(api.name for imp in self.imports for api in imp.entries)
        
        @property
        def checksum(self):
            return self._parsed.optional_header.computed_checksum
        
        @property
        def entrypoint(self):
            return self._parsed.rva_to_offset(self._parsed.optional_header.addressof_entrypoint)
        
        @property
        def entrypoint_section(self):
            return PESection(self._parsed.section_from_rva(self._parsed.optional_header.addressof_entrypoint), self)
        
        @property
        def iat(self):
            return self._parsed.data_directory(lief.PE.DataDirectory.TYPES.IMPORT_TABLE)
        
        @property
        def imported_dlls(self):
            return sorted(dll.name for dll in self.imports)
        
        def sections_with_slack_space(self, length=1):
            return [s for s in self if s.size - len(s.content) >= length]
        
        def sections_with_slack_space_entry_jump(self, offset=0):
            return self.sections_with_slack_space(6 + offset)
    
    PE.__name__ = "PE"
    PE.SECTION_CHARACTERISTICS = sec_chars
    PE.SECTION_TYPES = sec_types
    return PE
lazy_load_object("PE", __init_pe)

