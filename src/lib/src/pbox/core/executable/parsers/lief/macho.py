# -*- coding: UTF-8 -*-
from .__common__ import *


__all__ = ["MachO"]


_FLAGS = {
    'DEBUG_INFO':          "D",
    'EXT_RELOC':           "E",
    'SOME_INSTRUCTIONS':   "I",
    'LIVE_SUPPORT':        "L",
    'SELF_MODIFYING_CODE': "M",
    'NO_DEAD_STRIP':       "N",
    'LOC_RELOC':           "R",
    'STRIP_STATIC_SYMS':   "S",
    'NO_TOC':              "T",
    'PURE_INSTRUCTIONS':   "X",
}


def __init_macho():
    _p, _rn = cached_property, lambda sk: sk.lower()
    sec_flags = {k: v for k, v in getattr(lief.MachO.Section.FLAGS, "_member_map_").items()}
    sec_types = {k: v for k, v in getattr(lief.MachO.Section.TYPE, "_member_map_").items()}
    is_ = {f'is_{_rn(k)}': _make_property(k) for k in _FLAGS}
    MachOSection = get_part_class("MachOSection",
        flags="flags",
        flags_str=_p(lambda s: "".join(["", v][getattr(s, f"is_{_rn(k)}")] for k, v in _FLAGS.items())),
        is_code=lambda s: s.flags & 0x80000000 > 0 or s.flags & 0x00000400 > 0,
        is_data=lambda s: s.flags & 0x80000000 == 0 and s.flags & 0x00000400 == 0,
        raw_data_size=_p(lambda s: len(s.content)),
        virtual_size="size",
        FLAGS=sec_flags,
        TYPES=sec_types,
        **is_,
    )
    seg_flags = {k: v for k, v in getattr(lief.ELF.Segment.FLAGS, "_member_map_").items()}
    seg_types = {k: v for k, v in getattr(lief.ELF.Segment.TYPE, "_member_map_").items()}
    MachOSegment = get_part_class("MachOSegment",
        command="command",
        command_offset="command_offset",
        data="data",
        index="index",
        name="name",
        offset="file_offset",
        physical_address="physical_address",
        physical_size="size",
        FLAGS=seg_flags,
        TYPES=seg_types,
    )
    
    class MachO(Binary):
        checksum = 0
        DATA = "__data"
        TEXT = "__text"
        
        def __iter__(self):
            for s in self._parsed.sections:
                yield MachOSection(s, self)
        
        def _get_builder(self):
            return lief.MachO.Builder(self._parsed)
        
        @property
        def entrypoint(self):
            return self.virtual_address_to_offset(self._parsed.entrypoint)
        
        @property
        def entrypoint_section(self):
            return MachOSection(self.section_from_offset(self.entrypoint), self)
        
        @property
        def libraries(self):
            return [l.name for l in self._parsed.libraries]
        
        @property
        def sections(self):
            return [MachOSection(s, self) for s in self._parsed.sections]
        
        @property
        def segments(self):
            return [MachOSegment(s, self) for s in self._parsed.segments]
    
    MachO.__name__ = "MachO"
    MachO.SECTION_FLAGS = sec_flags
    MachO.SECTION_TYPES = sec_types
    MachO.SEGMENT_FLAGS = seg_flags
    MachO.SEGMENT_TYPES = seg_types
    return MachO
lazy_load_object("MachO", __init_macho)

