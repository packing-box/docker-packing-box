# -*- coding: UTF-8 -*-
from functools import cached_property
from magic import from_file
from tinyscript import classproperty, hashlib, shutil
from tinyscript.helpers import is_filetype, Path

from .__common__ import expand_categories
from ..learning.features import *


__all__ = ["expand_categories", "Executable"]


SIGNATURES = {
    '^Mach-O 32-bit ':                         "Mach-O32",
    '^Mach-O 64-bit ':                         "Mach-O64",
    '^Mach-O universal binary ':               "Mach-Ou",
    '^MS-DOS executable\s*':                   "MSDOS",
    '^PE32\+? executable (.+?)\.Net assembly': ".NET",
    '^PE32 executable ':                       "PE32",
    '^PE32\+ executable ':                     "PE64",
    '^(set[gu]id )?ELF 32-bit ':               "ELF32",
    '^(set[gu]id )?ELF 64-bit ':               "ELF64",
}


class Executable(Path):
    """ Executable abstraction. """
    def __init__(self):
        self._features = Features(self.category)
    
    def copy(self):
        shutil.copy(str(self), str(self.destination))
        self.destination.chmod(0o777)
    
    @cached_property
    def category(self):
        best_fmt, l = None, 0
        for ftype, fmt in SIGNATURES.items():
            if len(ftype) > l and is_filetype(str(self), ftype):
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def data(self):
        data = {}
        for name, func in self._features.items():
            r = func(str(self))
            if isinstance(r, dict):
                data.update(r)
            else:
                data[name] = r
        return data
    
    @cached_property
    def destination(self):
        return self.dataset.joinpath("files", self.hash)
    
    @cached_property
    def filetype(self):
        return from_file(str(self))
    
    @cached_property
    def filetype(self):
        return from_file(str(self))
    
    @cached_property
    def hash(self):
        return hashlib.sha256_file(str(self))
    
    @classproperty
    def features(cls):
        return self._features.descriptions

