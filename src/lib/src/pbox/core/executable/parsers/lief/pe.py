# -*- coding: UTF-8 -*-
from .__common__ import *


__all__ = ["PE"]


_FLAGS = {
    'MEM_DISCARDABLE': "D",
    'MEM_READ':        "R",
    'MEM_SHARED':      "S",
    'MEM_WRITE':       "W",
    'MEM_EXECUTE':     "X",
}
_FLAG_RENAME = {
    'MEM_EXECUTE': "executable",
    'MEM_READ':    "readable",
    'MEM_WRITE':   "writable",
}


def __init_pe():
    _p, _rn = cached_property, lambda sk: _FLAG_RENAME.get(sk, sk).lower().replace("mem_", "").replace("_", "")
    sec_chars = {k: v for k, v in getattr(lief.PE.Section.CHARACTERISTICS, "_member_map_").items()}
    sec_types = {k: v for k, v in getattr(lief.PE.SECTION_TYPES, "_member_map_").items()}
    is_ = {f'is_{_rn(k)}': _make_property(k) for k in _FLAGS}
    PESection = get_part_class("PESection",
        characteristics="characteristics",
        flags="characteristics",
        flags_str=_p(lambda s: "".join(["", v][getattr(s, f"is_{_rn(k)}")] for k, v in _FLAGS.items())),
        is_code=_p(lambda s: s.flags & 0x20 > 0),  # IMAGE_SCN_CNT_CODE ; IMAGE_SCN_MEM_EXECUTE (0x20000000) not
                                                   #  considered as the flag could be changed at runtime to make the
                                                   #  section executable
        is_data=lambda s: _p(s.flags & 0x40 > 0 or s.flags & 0x80 > 0),  # IMAGE_SCN_CNT_(UN)INITIALIZED_DATA
        raw_data_size="size",
        real_name="name",
        virtual_size="virtual_size",
        CHARACTERISTICS=sec_chars,
        FLAGS=sec_chars,
        TYPES=sec_types,
        **is_,
    )
    
    class PE(Binary):
        _build_cfg = BuildConfig("dos_stub", "imports", "overlay", "relocations", "resources", "tls", "patch_imports")
        DATA = ".data"
        TEXT = ".text"
        
        def __iter__(self):
            for s in self._parsed.sections:
                s = PESection(s, self)
                if hasattr(self, "_real_section_names"):
                    s.real_name = self.real_section_names.get(s.name, s.name)
                yield s
        
        def _get_builder(self):
            builder = lief.PE.Builder(self._parsed)
            for instruction, flag in list(self._build_cfg.items()):
                getattr(builder, instruction if instruction.startswith("patch") else f"build_{instruction}")(flag)
                del self._build_cfg[instruction]
            builder.build_overlay(False)  # build_overlay(True) fails when adding a section to the binary
            return builder
        
        @property
        def checksum(self):
            return self._parsed.optional_header.checksum
        
        @property
        def data_directories(self):
            class DataDirectory(GetItemMixin):
                __slots__ = ["has_section", "rva", "section", "size", "type"]
                def __init__(self2, dd):
                    for attr in self2.__slots__:
                        setattr(self2, attr, getattr(dd, attr))
            for dd in self._parsed.data_directories:
                yield DataDirectory(dd)
        
        @property
        def entrypoint(self):
            return self._parsed.rva_to_offset(self._parsed.optional_header.addressof_entrypoint)
        
        @property
        def entrypoint_section(self):
            return PESection(self._parsed.section_from_rva(self._parsed.optional_header.addressof_entrypoint), self)
        
        @property
        def iat(self):
            class IAT(GetItemMixin):
                __slots__ = ["rva", "section", "size", "type"]
                def __init__(self2):
                    iat = self._parsed.data_directory(lief.PE.DataDirectory.TYPES.IMPORT_TABLE)
                    for attr in self2.__slots__:
                        if attr == "section":
                            self2.section = PESection(iat.section, self)
                            continue
                        setattr(self2, attr, getattr(iat, attr))
            return IAT()
        
        @property
        def imported_apis(self):
            return {(dll.name, api.name) for dll in self.imports for api in dll.entries}
        
        @property
        def imported_dlls(self):
            return {dll.name for dll in self._parsed.imports}
        
        @property
        def imports(self):
            class Import(GetItemMixin):
                __slots__ = ["directory", "entries", "forwarder_chain", "iat_directory", "import_address_table_rva",
                             "import_lookup_table_rva", "name", "timedatestamp"]
                def __init__(self2, lief_import):
                    for attr in self2.__slots__:
                        setattr(self2, attr, getattr(lief_import, attr))
            for imp in self._parsed.imports:
                yield Import(imp)
        
        @property
        def machine(self):
            return self._parsed.header.machine.value
        
        @property
        def sections(self):
            return list(s for s in self)
    
    PE.__name__ = "PE"
    PE.SECTION_CHARACTERISTICS = sec_chars
    PE.SECTION_TYPES = sec_types
    return PE
lazy_load_object("PE", __init_pe)

