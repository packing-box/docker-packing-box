# -*- coding: UTF-8 -*-
from tinyscript import os, ts

from ..__common__ import *


__all__ = ["get_section_class", "lief", "Binary", "BuildConfig"]


def __init(lief):
    errors = config['lief_errors']
    def _lief_parse(target, *args, **kwargs):
        target = ts.Path(target, expand=True)
        if not target.exists():
            raise OSError("Target binary does not exist")
        if not errors:
            # capture the stderr messages from LIEF
            tmp_fd, null_fd = os.dup(2), os.open(os.devnull, os.O_RDWR)
            os.dup2(null_fd, 2)
        binary = lief._parse(str(target))
        if not errors:
            # restore stderr
            os.dup2(tmp_fd, 2)
            os.close(null_fd)
        if binary is None:
            raise OSError("Unknown format")
        return binary
    lief._parse, lief.parse = lief.parse, _lief_parse
    # monkey-patch header-related classes
    def __getitem__(self, name):
        if not name.startswith("_"):
            return getattr(self, name)
        raise KeyNotAllowedError(name)
    lief._lief.PE.Header.__getitem__ = __getitem__
    lief._lief.PE.OptionalHeader.__getitem__ = __getitem__
    return lief
ts.lazy_load_module("lief", postload=__init)


class Binary(AbstractParsedExecutable):
    def __new__(cls, path, *args, **kwargs):
        self = super().__new__(cls)
        self._parsed = lief.parse(str(path))
        if self._parsed is not None and cls.__name__.upper() == self._parsed.format.name:
            return self
    
    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self, "_parsed") and hasattr(self._parsed, name):
                return getattr(self._parsed, name)
            raise
    
    def build(self):
        builder = self._get_builder()
        builder.build()
        builder.write(self.name)
        with open(self.name, 'ab') as f:
            f.write(bytes(self.overlay))


class BuildConfig(dict):
    def toggle(self, **kwargs):
        for name, boolean in kwargs.items():
            self[name] = self.get(name, True) & boolean

