# -*- coding: UTF-8 -*-
from .__common__ import *


__all__ = ["ELF"]


_FLAGS = {
    'ALLOC':            "A",
    'COMPRESSED':       "C",
    'EXCLUDE':          "E",
    'GROUP':            "G",
    'INFO_LINK':        "I",
    'LINK_ORDER':       "L",
    'MERGE':            "M",
    'NONE':             "N",
    'OS_NONCONFORMING': "O",
    'STRINGS':          "S",
    'TLS':              "T",
    'WRITE':            "W",
    'EXECINSTR':        "X",
}
_FLAG_RENAME = {
    'ALLOC':     "allocated",
    'EXECINSTR': "executable",
    'WRITE':     "writable",
}


def __init_elf():
    _p, _rn = cached_property, lambda sk: _FLAG_RENAME.get(sk, sk).lower().replace("_", "")
    sec_flags = {k: v for k, v in getattr(lief.ELF.Section.FLAGS, "_member_map_").items()}
    sec_types = {k: v for k, v in getattr(lief.ELF.Section.TYPE, "_member_map_").items()}
    is_ = {f'is_{_rn(k)}': _make_property(k) for k in _FLAGS}
    ELFSection = get_part_class("ELFSection",
        flags="flags",
        flags_str=_p(lambda s: "".join(["", v][getattr(s, f"is_{_rn(k)}")] for k, v in _FLAGS.items())),
        is_code=_p(lambda s: s.flags & 0x4 > 0),
        is_data=_p(lambda s: s.flags & 0x1 > 0 and s.flags & 0x2 > 0 and s.flags & 0x4 == 0),
        raw_data_size=_p(lambda s: len(s.content)),
        virtual_size="size",
        FLAGS=sec_flags,
        TYPES=sec_types,
        **is_,
    )
    seg_flags = {k: v for k, v in getattr(lief.ELF.Segment.FLAGS, "_member_map_").items()}
    seg_types = {k: v for k, v in getattr(lief.ELF.Segment.TYPE, "_member_map_").items()}
    ELFSegment = get_part_class("ELFSegment",
        alignment="alignment",
        offset="file_offset",
        physical_address="physical_address",
        physical_size="physical_size",
        type="type",
        FLAGS=seg_flags,
        TYPES=seg_types,
    )
    
    class ELF(Binary):
        checksum = 0
        DATA = ".data"
        TEXT = ".text"
        
        def __iter__(self):
            for s in self._parsed.sections:
                yield ELFSection(s, self)
        
        def _get_builder(self):
            return lief.ELF.Builder(self._parsed)
        
        @property
        def entrypoint(self):
            return self.virtual_address_to_offset(self._parsed.entrypoint)
        
        @property
        def entrypoint_section(self):
            return ELFSection(self.section_from_offset(self._parsed.entrypoint), self)
        
        @property
        def segments(self):
            return [ELFSegment(s, self) for s in self._parsed.segments]
    
    ELF.__name__ = "ELF"
    ELF.SECTION_FLAGS = sec_flags
    ELF.SECTION_TYPES = sec_types
    ELF.SEGMENT_FLAGS = seg_flags
    ELF.SEGMENT_TYPES = seg_types
    return ELF
lazy_load_object("ELF", __init_elf)

