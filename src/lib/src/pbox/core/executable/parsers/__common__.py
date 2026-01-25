# -*- coding: UTF-8 -*-
import weakref
from abc import ABC, abstractmethod
from tinyscript import functools, re
from tinyscript.helpers import ensure_str, execute, is_generator, zeropad

from ....helpers.data import get_data
from ....helpers.mixins import *

lazy_load_module("bintropy")


__all__ = ["get_part_class", "supported_parsers", "AbstractParsedExecutable", "CustomReprMixin", "GetItemMixin"]


_ATTR_REGEX = re.compile(r"(_address|_?alignment|characteristics|flags|index|_?offset|_?size)$", re.I)
_DEF_REGEX  = re.compile(r"^$")
_GETI_REGEX = re.compile(r"^(?!has_)(|[a-z]+_)(configuration|header)$", re.I)
_STR_REGEX  = re.compile(r"^.*_str$", re.I)

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


class NullSection(GetItemMixin):
    def __getattr__(self, name):
        # do not consider returning "" for 'name' as it would mean empty section name
        return "" if _STR_REGEX.search(name) else -1 if _ATTR_REGEX.search(name) else None


class PartsList(list):
    def __contains__(self, name_or_part):
        if isinstance(name_or_part, str):
            return name_or_part in map(lambda x: x.name, self)
        elif any(name_or_part.__class__.__name__.endswith(x) for x in ["Section", "Segment"]) and \
             hasattr(name_or_part, "name"):
            return name_or_part.name in map(lambda x: x.name, self)
        return False


class AbstractParsedExecutable(ABC, CustomReprMixin, GetItemMixin):
    def __getitem__(self, name):
        try:
            v = super().__getitem__(name)
            if (_GETI_REGEX.search(name) or getattr(self, "_getitem_regex", _DEF_REGEX).search(name)) and \
               not hasattr(v, "__getitem__"):
                setattr(v.__class__, "__getitem__", GetItemMixin.__getitem__.__get__(v, v.__class__))
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
            w = len(c := _rb(getattr(getattr(self, "_parsed", self), "overlay", b"")))
            r += (bintropy.entropy(c, blocksize, ignore_half_block_zeros)[1] or 0.) * w
            t += w
        return r / (t or 1)
    
    def average_entropy_k_top_bytes(self, k=100, max_entropy=False):
        n, e = 0, 0
        for s in self.sections:
            e += (e_s := s.entropy_k_top_bytes(k, max_entropy))
            if e_s > 0.:
                n += 1
        return e / n
    
    def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True,
                      ignore_overlay=False):
        return bintropy.entropy(_rb(self.content if ignore_overlay else self.bytes), blocksize, ignore_half_block_zeros,
                                ignore_half_block_same_byte)
    
    def block_entropy_per_section(self, blocksize=256, ignore_half_block_zeros=True, ignore_half_block_same_byte=True):
        return {getattr(s, "real_name", s.name): s.block_entropy(blocksize, ignore_half_block_zeros,
                                                                 ignore_half_block_same_byte) for s in self}
    
    @cached_result
    def bytes_after_entrypoint(self, n):
        return zeropad(n)([_ for _ in _rb(self.entrypoint_section.content[:n])])
    
    def entropy_sections(self, sections):
        if sections is not None and not isinstance(sections, list):
            sections = [sections]
        if sections is None or len(sections) == 0:
            return
        return bintropy.entropy(b"".join(_rb((self.section(s) if isinstance(s, str) else s).content) for s in sections))
    
    def fragments(self, fragment_size=1024, n=3, skip_header=True, from_end=False, ignore_overlay=True, pad=False):
        o = self.size_of_header if not from_end and skip_header else 0
        d, fs = self.content if ignore_overlay else self.bytes, fragment_size
        for i in (range(-fs, -1, -fs) if from_end else range(0, n * fs, fs)):
            yield f + b"\x00" * (fs - l) if pad and (l := len(f := d[o+i:o+i+fs])) < fs else f
    
    def has_section(self, name_or_section):
        return getattr(name, "name", name) in self.section_names()
    
    def lrc_fragments(self, fragment_size=1024, n=3, skip_header=True, from_end=False, ignore_overlay=True, pad=False):
        for f in self.fragments(fragment_size=fragment_size, n=n, skip_header=skip_header, from_end=from_end,
                                ignore_overlay=ignore_overlay, pad=pad):
            yield bytes([(f[i] + f[i+1]) & 0xFF for i in range(len(f)-1)])
    
    def modify(self, modifier, **kw):
        modifier(self.parsed, **kw)
        self.build()
    
    def section(self, section, original=False, null=False):
        if isinstance(section, (bytes, str)):
            name = ensure_str(section)
            if original:
                for s1, s2 in zip(self, self._parsed.sections):
                    if ensure_str(s1.name) == name:
                        return s2                    
            else:
                for s in self:
                    real_name = getattr(self, "real_section_names", {}).get(s.name, s.name)
                    if ensure_str(s.name) == name or ensure_str(real_name) == name:
                        if hasattr(s, "real_name"):
                            s.real_name = real_name
                        return s
            if null:
                return NullSection()
            raise ValueError(f"no section named '{name}'")
        elif isinstance(section, GetItemMixin):
            if original:
                try:
                    return section._original
                except AttributeError:
                    for s1, s2 in zip(self, self._parsed.sections):
                        if ensure_str(s1.name) == ensure_str(section.name):
                            return s2
                # should not happen ; this would mean that the input section does not come from the current executable
                raise ValueError(f"no section named '{section.name}'")
            else:
                if hasattr(section, "real_name"):
                    section.real_name = self.real_section_names.get(section.name, section.name)
                return section
        elif hasattr(section, "name"):
            return section
        raise ValueError(".section(...) only supports a section name or a parsed section object as input")
    
    def section_names(self, *sections):
        return [s if isinstance(s, str) else s.name for s in (sections or self)]
    
    def sections_average_entropy(self, *sections):
        return sum(e := [s.entropy for s in (sections or self)]) / len(e)
    
    @property
    def _section_class(self):
        try:
            return type(self.sections[0])
        except IndexError:
            raise ParserError("Could not determine original section class")
    
    @property
    def code_sections(self):
        return PartsList(s for s in self if s.is_code)
    
    @property
    def content(self):
        return self.path.bytes[:-l] if (l := len(self._parsed.overlay.tobytes())) > 0 else self.path.bytes
    
    @property
    def data_section(self):
        return self.section(self.DATA, null=True)
    
    @property
    def data_sections(self):
        return PartsList(s for s in self if s.is_data)
    
    @property
    def empty_name_sections(self):
        return PartsList(s for s in self if _rn(s) == "")
    
    @property
    def format(self):
        return self.path.format
    
    @property
    def has_slack_space(self):
        return any(s.slack_space > 0 for s in self)
    
    @property
    def known_packer_sections(self):
        d = get_data(self.format)['COMMON_PACKER_SECTION_NAMES']
        return PartsList(s for s in self if _rn(s) in d)
    
    @property
    def last_section(self):
        try:
            return [s for s in self][-1]
        except IndexError:
            return NullSection()
    
    @property
    def non_standard_sections(self):
        d = [""] + get_data(self.format)['STANDARD_SECTION_NAMES']
        return PartsList(s for s in self if _rn(s) not in d)
    
    @property
    def real_section_names(self):
        """ This only applies to PE as section names are limited to 8 characters for image files ; when using longer
             names, they are mapped into a string table that 'objdump' can read to recover the real section names. """
        if self.path.group != "PE":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'real_section_names'")
        if not hasattr(self, "_real_section_names"):
            def _decode(pe, name, fh):
                if not name.startswith("/"):
                    return name
                string_table_offset = pe.header.pointerto_symbol_table + pe.header.numberof_symbols * 18
                fh.seek(string_table_offset + int(name[1:]))
                # read the null-terminated string from the file
                return b"".join(iter(lambda: (b := fh.read(1)) and b != b'' and b or b'\x00', b'\x00')).decode("utf-8", errors="ignore")
            # start parsing section names
            names = [ensure_str(s.name) for s in self]
            from re import match
            if all(match(r"/\d+$", n) is None for n in names):
                self._real_section_names = {}
                return self._real_section_names
            real_names = []
            with open(self.path, "rb") as f:
                for encoded_name in names:
                    real_name = _decode(self, encoded_name, f)
                    real_names.append(real_name)
            self._real_section_names = {n: rn for n, rn in zip(names, real_names) if match(r"/\d+$", n)}
        return self._real_section_names
    
    @property
    def sections(self):
        return list(s for s in self)
    
    @property
    def standard_sections(self):
        d = [""] + get_data(self.format)['STANDARD_SECTION_NAMES']
        return PartsList(s for s in self if _rn(s) in d)
    
    @property
    def text_section(self):
        return self.section(self.TEXT, null=True)
    
    @property
    def unknown_sections(self):
        d = [""] + get_data(self.format)['STANDARD_SECTION_NAMES'] + \
                   get_data(self.format)['COMMON_PACKER_SECTION_NAMES']
        return PartsList(s for s in self if _rn(s) not in d)
    
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


def get_part_class(clsname, **mapping):
    slots = DEFAULT_SECTION_SLOTS[:] if clsname.endswith("Section") else DEFAULT_SEGMENT_SLOTS[:]
    for attr in mapping.keys():
        if attr not in slots:
            slots.append(attr)
    
    class AbstractPart(CustomReprMixin, GetItemMixin):
        """ Abstract representation of a binary's section or segment.
        
        NB: Simple namespace to hold section information. Referencing a a native class (e.g. lief.PE.Section) directly
             is dangerous, because it can be modified by alterations, which will break things. Also, if different
             parsers are used in subsequent alterations, a common format is required.
        """
        __slots__ = ["binary"] + slots
        _instances = weakref.WeakValueDictionary()
        
        def __new__(cls, parent, name):
            key = (id(parent), name)
            if key in cls._instances:
                return cls._instances[key]
            cls._instances[key] = i = super().__new__(cls)
            return i
        
        def __init__(self, name, binary=None):
            for attr in self.__slots__:
                if attr == "binary":
                    self.binary = binary
                    try:
                        self._original = self.binary.section(name, original=True)
                    except:
                        print("DEBUG", name)
                        raise
                    continue
                value = mapping.get(attr, attr)
                if isinstance(value, (type(lambda: 0), cached_property)):
                    continue
                if isinstance(value, str):
                    tmp = name
                    for token in value.split("."):
                        tmp = getattr(tmp, token, None)
                    value = tmp
                setattr(self, attr, value)
        
        def __len__(self):
            return self.size
        
        def block_entropy(self, blocksize=256, ignore_half_block_zeros=False, ignore_half_block_same_byte=True):
            return bintropy.entropy(_rb(self.content), blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)
        
        def entropy_k_top_bytes(self, k=100, max_entropy=False):
            from ....helpers.utils import entropy
            return entropy(self.content, k, max_entropy)
        
        @property
        def bytes(self):
            return _rb(self.content)
        
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
            std_field = "standard_" + ["sections", "segments"][self.__class__.__name__.endswith('Segment')]
            return (sn := lambda s: getattr(s, "real_name", s.name))(self) in map(sn, getattr(self.binary, std_field))
        
        @property
        def portability(self):
            return 1.
        
        @property
        def slack_space(self):
            return max(0, getattr(self, "physical_size", getattr(self, "raw_data_size")) - self.virtual_size)
        
        def sections_with_slack_space(self, length=1):
            return {s for s in self if s.virtual_size - s.raw_data_size >= length}
        
        def sections_with_slack_space_entry_jump(self, offset=0):
            return self.sections_with_slack_space(6 + offset)
    
    for attr, value in mapping.items():
        if isinstance(value, cached_property):
            setattr(AbstractPart, attr, value)
            getattr(AbstractPart, attr).__set_name__(AbstractPart, attr)
        elif isinstance(value, type(lambda: 0)):
            setattr(AbstractPart, attr, value)
    AbstractPart.__name__ = clsname
    return AbstractPart

