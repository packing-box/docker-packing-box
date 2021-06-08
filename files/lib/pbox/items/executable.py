# -*- coding: UTF-8 -*-
from datetime import datetime
from functools import cached_property
from magic import from_file
from tinyscript import classproperty, hashlib, shutil
from tinyscript.helpers import is_filetype, Path

from ..learning.features import *


__all__ = ["Executable"]


# NB: the best signature matched is the longest
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
    _features = {}

    def __new__(cls, *parts, **kwargs):
        ds, h = kwargs.pop('dataset', None), kwargs.pop('data', None)
        opt = {}
        for k in ["category", "data", "filetype", "hash"]:
            opt[k] = kwargs.pop(k, None)
        if len(parts) == 0 and ds and h:
            parts = (ds.files.joinpath(h), )
        elif len(parts) == 0 and not ds:
            raise ValueError("Cannot determine executable's path")
        self = super(Executable, cls).__new__(cls, *parts, **kwargs)
        self.__hash = h #FIXME: use self.attributes
        self.__hash = kwargs.pop('hash', None)
        self.dataset = ds
        return self
    
    def __getattribute__(self, name):
        if name == "_features":
            fset = Executable._features.get(self.category)
            if fset is None:
                Executable._features[self.category] = fset = Features(self.category)
            return fset
        return super(Executable, self).__getattribute__(name)
    
    def copy(self):
        if str(self) != str(self.destination):
            shutil.copy(str(self), str(self.destination))
            self.destination.chmod(0o777)
    
    @property
    def attributes(self):
        return {n: getattr(self, n) for n in ["category", "ctime", "data", "filetype", "hash", "mtime"]} #FIXME
    
    @cached_property
    def category(self):
        best_fmt, l = None, 0
        for ftype, fmt in SIGNATURES.items():
            if len(ftype) > l and is_filetype(str(self), ftype):
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def ctime(self):
        return datetime.fromtimestamp(self.stat().st_ctime)
    
    @cached_property
    def data(self):
        if self.__data is not None:
            return self.__data
        data = {}
        for name, func in self._features.items():
            r = func(self)
            if isinstance(r, dict):
                data.update(r)
            else:
                data[name] = r
        return data
    
    @cached_property
    def destination(self):
        return self.dataset._file(self.hash, hash=self.hash, data=self.data)
    
    @property
    def features(self):
        return {n: FEATURE_DESCRIPTIONS.get(n, "") for n in self.data.keys()}
    
    @cached_property
    def filetype(self):
        try:
            return from_file(str(self))
        except OSError:
            return
    
    @cached_property
    def hash(self):
        return self.__hash or hashlib.sha256_file(str(self))
    
    @cached_property
    def mtime(self):
        return datetime.fromtimestamp(self.stat().st_mtime)
    
    @property
    def path(self):
        return Path(self.dataset._names[self.hash])

