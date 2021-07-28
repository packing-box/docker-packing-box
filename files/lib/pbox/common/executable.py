# -*- coding: UTF-8 -*-
from datetime import datetime
from functools import cached_property
from magic import from_file
from tinyscript import classproperty, hashlib, shutil
from tinyscript.helpers import is_filetype, Path


__all__ = ["Executable"]


class Executable(Path):
    """ Executable abstraction.
    
    Can be initialized in two different ways:
    (1) orphan (not dataset-bound)
    (2) dataset-bound (its data is used to populate attributes based on the executable's hash)
    """
    FIELDS = ["realpath", "category", "filetype", "size", "ctime", "mtime"]
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
    _features = {}

    def __new__(cls, *parts, **kwargs):
        self = parts[0] if len(parts) == 1 and isinstance(parts[0], Executable) else \
               super(Executable, cls).__new__(cls, *parts, **kwargs)
        try:
            ds, h = kwargs.pop('dataset'), kwargs.pop('hash', self.basename)
            self._dataset = ds
            d = ds._data[ds._data.hash == h].iloc[0].to_dict()
            self = super(Executable, cls).__new__(cls, ds.files.joinpath(h), **kwargs)
            for a, v in d.items():
                if a in Executable.FIELDS + ["hash", "label"]:
                    setattr(self, a, v)
        except (AttributeError, IndexError, KeyError):
            pass
        self.label = kwargs.pop('label', getattr(self, "label", None))
        return self
    
    def copy(self):
        if str(self) != str(self.destination) and not self.destination.exists():
            try:
                shutil.copy(str(self), str(self.destination))
            except:
                print(self, self.destination)
                raise
            self.destination.chmod(0o777)
    
    @property
    def metadata(self):
        return {n: getattr(self, n) for n in Executable.FIELDS}
    
    @cached_property
    def category(self):
        best_fmt, l = None, 0
        for ftype, fmt in Executable.SIGNATURES.items():
            if len(ftype) > l and is_filetype(str(self), ftype):
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def ctime(self):
        return datetime.fromtimestamp(self.stat().st_ctime)
    
    @cached_property
    def destination(self):
        try:
            return self._dataset.files.joinpath(self.hash)
        except (AttributeError, TypeError):
            raise ValueError("No 'Dataset' instance bound")
    
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
        return str(self)
    
    @cached_property
    def size(self):
        return super(Executable, self).size

