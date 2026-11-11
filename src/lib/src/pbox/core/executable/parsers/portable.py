# -*- coding: UTF-8 -*-
""" Parser based on pbox.portable, the dependency-free code that the tools exported from trained models embed.

Select it with the 'pe_parser' option (e.g. 'packing-box config --pe-parser portable') to compute features with exactly
 the code of the exported tools ; the parity tests compare it with the default (LIEF-based) parser.

Only PE is supported. The parsed binary cannot be modified (no alteration) nor disassembled.
"""
from ....helpers.data import get_data
from ....portable.parsers.pe import PE as _PortablePE, PEError
from .__common__ import AbstractParsedExecutable


__all__ = ["PE", "PEError"]


# members of AbstractParsedExecutable that the feature expressions use and that shall come from the portable code
#  (otherwise, the generic implementations of AbstractParsedExecutable would be used instead)
_DELEGATED = ["average_block_entropy_per_section", "block_entropy", "bytes_after_entrypoint", "empty_name_sections",
              "known_packer_sections", "non_standard_sections", "real_section_names", "standard_sections"]


class PE(AbstractParsedExecutable):
    """ pbox's view of a PE parsed by pbox.portable.parsers.pe.PE. """
    DATA = ".data"
    TEXT = ".text"

    def __init__(self, path):
        self._portable = _PortablePE(str(path))
        self._data_set = False

    def __getattr__(self, name):
        # attributes not defined by pbox's abstraction come from the portable parser
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._p, name)

    def __iter__(self):
        return iter(self._p.sections)

    @property
    def _p(self):
        """ The portable parser, with the section names of pbox's data once the path (hence the format) is known. """
        p = self.__dict__['_portable']
        if not self.__dict__.get('_data_set') and 'path' in self.__dict__:
            data = get_data(self.path.format)
            p.standard_section_names = list(data.get('STANDARD_SECTION_NAMES', []))
            p.common_packer_section_names = list(data.get('COMMON_PACKER_SECTION_NAMES', []))
            self._data_set = True
        return p

    @property
    def content(self):
        return self._p.data[:len(self._p.data) - len(self._p.overlay)]

    @property
    def entrypoint(self):
        return self._p.entrypoint

    @property
    def entrypoint_section(self):
        return self._p.entrypoint_section

    @property
    def overlay(self):
        return self._p.overlay

    @property
    def sections(self):
        return self._p.sections

    @property
    def size_of_header(self):
        return self._p.sizeof_headers


def _delegate(name):
    attr = getattr(_PortablePE, name)
    if isinstance(attr, property):
        return property(lambda self: getattr(self._p, name), doc=attr.__doc__)
    def _method(self, *args, **kwargs):
        return getattr(self._p, name)(*args, **kwargs)
    _method.__name__, _method.__doc__ = name, attr.__doc__
    return _method


for _name in _DELEGATED:
    setattr(PE, _name, _delegate(_name))
