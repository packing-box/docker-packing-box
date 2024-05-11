# -*- coding: UTF-8 -*-
from abc import ABC, abstractmethod
from tinyscript import functools
from tinyscript.helpers import ensure_str, execute, is_generator

from ....helpers.data import get_data
from ....helpers.mixins import *

lazy_load_module("bintropy")


__all__ = ["get_section_class", "supported_parsers", "AbstractParsedExecutable"]


_rb = lambda s: bytes(getattr(s, "tobytes", lambda: s)())
_rn = lambda s: ensure_str(getattr(s, "real_name", s.name)).split("\0")[0]


def supported_parsers(*parsers):
    def _wrapper(f):
        @functools.wraps(f)
        def _subwrapper(parsed, *args, **kwargs):
            if parsed.parser not in parsers:
                raise ParserError(f"parser '{parsed.parser}' not supported for alteration '{f.__name__.lstrip('_')}'")
            return f(parsed, *args, **kwargs)
        return _subwrapper
    return _wrapper


class AbstractParsedExecutable(ABC, CustomReprMixin, GetItemMixin):
    def __getitem__(self, name):
        try:
            v = super().__getitem__(name)
            if name.lower().endswith("header") and not hasattr(v, "__getitem__"):
                # if not done yet, patch Header class with a getitem method too (e.g. for getting exe['header']['...'])
                try:
                    setattr(v, "__getitem__", GetItemMixin.__getitem__.__get__(v, v.__class__))
                except AttributeError:
                    pass
        except AttributeError:
            if hasattr(self, "path") and hasattr(self.path, name):
                return getattr(self.path, name)
            raise
        return v
    
    def average_block_entropy_per_section(self, blocksize=256, ignore_half_block_zeros=True, overlay=True, raw=True):
        r, t = 0., 0
        for e, w in [(bintropy.entropy(_rb(s.content), blocksize, ignore_half_block_zeros),
                      len(_rb(s.content)) if raw else s.size) for s in self]:
            if e == .0 or e[1] in [.0, None]:
                continue
            r += e[1] * w
            t += w
        if overlay:
            # get overlay, first from the eventually-defined attribute "_parsed", otherwise from the current instance
            o = getattr(getattr(self, "_parsed", self), "overlay", b"")
            c = _rb(o)
            w = len(c)
            r += (bintropy.entropy(c, blocksize, ignore_half_block_zeros)[1] or 0.) * w
            t += w
        return r / (t or 1)
    
    def sections_average_entropy(self, *sections):
        e = [s.entropy for s in (sections or self)]
        return sum(e) / len(e)
    
    def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True):
        return bintropy.entropy(_rb(self.code), blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)
    
    def block_entropy_per_section(self, blocksize=256, ignore_half_block_zeros=True, ignore_half_block_same_byte=True):
        return {getattr(s, "real_name", s.name): s.block_entropy(blocksize, ignore_half_block_zeros,
                                                                 ignore_half_block_same_byte) for s in self}
    
    def modify(self, modifier, **kw):
        modifier(self.parsed, **kw)
        self.build()
    
    def section(self, section, original=False):
        if isinstance(section, (bytes, str)):
            name = ensure_str(section)
            if original:
                for s1, s2 in zip(self, self.sections):
                    if ensure_str(s1.name) == name:
                        return s2                    
            else:
                for s in self:
                    real_name = getattr(self, "real_section_names", {}).get(s.name, s.name)
                    if ensure_str(s.name) == name or ensure_str(real_name) == name:
                        if hasattr(s, "real_name"):
                            s.real_name = real_name
                        return s
            raise ValueError(f"no section named '{name}'")
        elif isinstance(section, GetItemMixin):
            if original:
                for s1, s2 in zip(self, self.sections):
                    if ensure_str(s1.name) == ensure_str(section.name):
                        return s2
                # should not happen ; this would mean that the input section does not come from the current executable
                raise ValueError(f"no section named '{section.name}'")
            else:
                if hasattr(section, "real_name"):
                    section.real_name = self.real_section_names.get(section.name, section.name)
                return section
        elif hasattr(section, "name"):
            return self.section(section.name, original)
        raise ValueError(".section(...) only supports a section name or a parsed section object as input")
    
    def section_names(self, *sections):
        sections = sections[0] if len(sections) == 1 and isinstance(sections, (list, tuple)) else sections
        return [s.name for s in (sections or self)]
    
    @property
    def _section_class(self):
        try:
            first = self.sections[0] if isinstance(self.sections, (list, tuple)) and len(self.sections) > 0 else \
                    next(self.sections) if is_generator(self.sections) else None
        except StopIteration:
            first = None
        if first is None:
            raise ParserError("Could not determine original section class")
        return type(first)
    
    @property
    def code(self):
        return self.path.bytes
    
    @property
    def empty_name_sections(self):
        return [s for s in self if _rn(s) == ""]
    
    @property
    def known_packer_sections(self):
        d = get_data(self.path.format)['COMMON_PACKER_SECTION_NAMES']
        return [s for s in self if _rn(s) in d]
    
    @property
    def non_standard_sections(self):
        d = [""] + get_data(self.path.format)['STANDARD_SECTION_NAMES']
        return [s for s in self if _rn(s) not in d]
    
    @property
    def real_section_names(self):
        """ This only applies to PE as section names are limited to 8 characters for image files ; when using longer
             names, they are mapped into a string table that 'objdump' can read to recover the real section names. """
        if self.path.group != "PE":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'real_section_names'")
        if not hasattr(self, "_real_section_names"):
            names = [ensure_str(s.name) for s in self]
            from re import match
            if all(match(r"/\d+$", n) is None for n in names):
                self._real_section_names = {}
                return self._real_section_names
            real_names = []
            out, _ = execute(["objdump", "-h", str(self.path)])
            for l in out.decode("latin-1").split("\n"):
                m = match(r"\s+\d+\s(.*?)\s+", l)
                if m:
                    real_names.append(m.group(1))
            self._real_section_names = {n: rn for n, rn in zip(names, real_names) if match(r"/\d+$", n)}
        return self._real_section_names
    
    @property
    def standard_sections(self):
        d = [""] + get_data(self.path.format)['STANDARD_SECTION_NAMES']
        return [s for s in self if _rn(s) in d]
    
    # ------------------------------------------- mandatory concrete methods -------------------------------------------
    @abstractmethod
    def __iter__(self):
        pass
    
    # ------------------------------------------- optional concrete methods --------------------------------------------
    def build(self, **kw):
        raise NotImplementedError("build")
    
    def disassemble(self, **kw):
        raise NotImplementedError("disassemble")
    
    @property
    def checksum(self):
        raise NotImplementedError("checksum")


def get_section_class(name, **mapping):
    slots = DEFAULT_SECTION_SLOTS[:]
    for attr in mapping.keys():
        if attr not in slots:
            slots.append(attr)
    
    class AbstractSection(CustomReprMixin, GetItemMixin):
        """ Abstract representation of a binary's section.
        
        NB: Simple namespace to hold section information. Referencing a lief.PE.Section directly is dangerous, because
             it can be modified by alterations, which will break things. Also, if different parsers are used in
             subsequent alterations, a common format is required.
        """
        __slots__ = ["binary"] + slots
        
        def __init__(self, section, binary=None):
            for attr in self.__slots__:
                if attr == "binary":
                    self.binary = binary
                    continue
                value = mapping.get(attr, attr)
                if isinstance(value, (type(lambda: 0), cached_property)):
                    continue
                if isinstance(value, str):
                    tmp = section
                    for token in value.split("."):
                        tmp = getattr(tmp, token, None)
                    value = tmp
                setattr(self, attr, value)
        
        def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True):
            return bintropy.entropy(_rb(self.content), blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)
        
        @property
        def block_entropy_256B(self):
            return self.block_entropy(256, True)[1]
        
        @property
        def block_entropy_512B(self):
            return self.block_entropy(512, True)[1]
        
        @property
        def entropy(self):
            return bintropy.entropy(_rb(self.content))
        
        @property
        def is_standard(self):
            if self.binary is None:
                return
            sn = lambda s: getattr(s, "real_name", s.name)
            return sn(self) in map(sn, self.binary.standard_sections)
    
    for attr, value in mapping.items():
        if isinstance(value, cached_property):
            setattr(AbstractSection, attr, value)
            getattr(AbstractSection, attr).__set_name__(AbstractSection, attr)
        elif isinstance(value, type(lambda: 0)):
            setattr(AbstractSection, attr, value)
    AbstractSection.__name__ = name
    return AbstractSection

