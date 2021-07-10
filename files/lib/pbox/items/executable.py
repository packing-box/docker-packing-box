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
    """ Executable abstraction.
    
    Can be initialized in four different ways:
    (1) Executable instance provided as single element in 'parts', with 'dataset' kwarg set
    (2) Classical Path instance, with 'parts' and no dataset bound
    (3) Classical Path instance, with 'parts' and 'dataset' kwarg set for binding a parent Dataset instance
    (4) No 'parts' provided, but a Dataset instance and a hash are provided as kwargs
    """
    _features = {}

    def __new__(cls, *parts, **kwargs):
        ds = kwargs.pop('dataset', None)
        l = kwargs.pop('label', None)
        # cases 1 to 3
        if len(parts) > 0:
            e = parts[0]
            if len(parts) == 1 and isinstance(e, Executable) and ds == getattr(e, "dataset", None):
                return e
            self = super(Executable, cls).__new__(cls, *parts, **kwargs)
            # case 1: an Executable instance is given, i.e. from another dataset ; make self inherit its properties
            if len(parts) == 1 and isinstance(e, Executable):
                self.hash = e.destination.filename
                # otherwise, clone its cached properties
                for attr in ["category", "data", "filetype", "realpath"]:
                    setattr(self, attr, getattr(e, attr))
                l = e.label
            # case 3: a dataset is given, bind it and copy the input executable to dataset's files if relevant
            if ds:
                self.dataset = ds
                if self.absolute().is_under(ds.files):
                    self.hash = self.filename
                ds[self] = l
                ds._save()
                
        # case 4: get cached properties and data for the given hash from the bound dataset
        else:
            try:
                h = kwargs.pop('hash')
                d = ds[h, True]
            except (KeyError, TypeError):
                raise ValueError("When no 'parts' arg is provided, 'dataset' and 'hash' kwargs must be provided")
            self = super(Executable, cls).__new__(cls, ds.files.joinpath(h), **kwargs)
            self.dataset, self.hash, l, _ = ds, h, d.pop('label'), d.pop('hash')
            self.data = d
        self.label = l
        return self
    
    def __getattribute__(self, name):
        if name == "_features":
            fset = Executable._features.get(self.category)
            if fset is None:
                Executable._features[self.category] = fset = Features(self.category)
            return fset
        return super(Executable, self).__getattribute__(name)
    
    def copy(self):
        if str(self) != str(self.destination) and not self.destination.exists():
            shutil.copy(str(self), str(self.destination))
            self.destination.chmod(0o777)
    
    @property
    def attributes(self):
        return {n: getattr(self, n) for n in ["category", "ctime", "data", "filetype", "hash", "mtime"]}
    
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
        try:
            return self.dataset.files.joinpath(self.hash)
        except (AttributeError, TypeError):
            raise ValueError("No 'Dataset' instance bound")
    
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
        return hashlib.sha256_file(str(self))
    
    @cached_property
    def mtime(self):
        return datetime.fromtimestamp(self.stat().st_mtime)
    
    @cached_property
    def realpath(self):
        try:
            return Path(self.dataset._names[self.hash])
        except (AttributeError, TypeError):
            raise ValueError("No 'Dataset' instance bound")

