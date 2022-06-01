# -*- coding: UTF-8 -*-
from datetime import datetime
from functools import cached_property
from magic import from_file
from tinyscript import classproperty, hashlib, shutil
from tinyscript.helpers import is_filetype, Path


__all__ = ["Executable"]


class Executable(Path):
    """ Executable abstraction.
    
    Can be initialized in many different ways:
    (1) orphan executable (not coming from a dataset)
    (2) dataset-bound (its data is used to populate attributes based on the executable's hash)
    (3) dataset-bound, with a new destination dataset
    """
    FIELDS = ["realpath", "filetype", "format", "size", "ctime", "mtime"]
    HASH = "sha256"  # possible values: hashlib.algorithms_available
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
        data, fields = None, ["hash", "label"] + Executable.FIELDS
        ds1, ds2 = kwargs.pop('dataset', None), kwargs.pop('dataset2', None)
        if len(parts) == 1:
            e = parts[0]
            # if reinstantiating an Executable instance, simply immediately return it
            if isinstance(e, Executable):
                return e
            # this case arises when a series is passed from Pandas' .itertuples() ; this produces an orphan executable
            if all(hasattr(e, f) for f in fields) and hasattr(e, "_fields"):
                try:
                    dest = ds1.files.joinpath(e.hash)
                except AttributeError:  # 'NoneType|FilelessDataset' object has no attribute 'files'
                    dest = e.realpath
                self = super(Executable, cls).__new__(cls, dest, **kwargs)
                for f in fields:
                    setattr(self, f, getattr(e, f))
                if ds1:
                    self._dataset = ds1
                    try:
                        self._destination = self._dataset.files.joinpath(self.hash)
                    except AttributeError:  # 'FilelessDataset' object has no attribute 'files'
                        pass
                return self
        self = super(Executable, cls).__new__(cls, *parts, **kwargs)
        if ds1 is not None:
            # case (2)
            h = kwargs.pop('hash', self.basename)
            if ds1._files:
                exe = ds1.files.joinpath(h)
                if exe.is_file():
                    self = super(Executable, cls).__new__(cls, str(exe), **kwargs)
                    self.hash = h  # avoid hash recomputation
                    self._destination = ds1.files.joinpath(self.hash)
            self._dataset = ds1
            try:
                for a, v in ds1._data[ds1._data.hash == h].iloc[0].to_dict().items():
                    if a in Executable.FIELDS + ["hash", "label"]:
                        setattr(self, a, v)
            except AttributeError:
                pass  # this occurs when the dataset is still empty, therefore holding no 'hash' column
            except IndexError:
                pass  # this occurs when the executable did not exist in the dataset yet
            # case (3)
            if ds2 is not None and ds2._files:
                self._destination = ds2.files.joinpath(h)
        self.label = kwargs.pop('label', getattr(self, "label", None))
        return self
    
    def copy(self, extension=False):
        dest = Path(str(self.destination) + ["", self.extension.lower()][extension])
        if str(self) != dest and not dest.exists():
            try:  # copy file with its attributes and metadata
                shutil.copy2(str(self), str(dest))
            except:
                raise
            dest.chmod(0o777)
            return dest
    
    @property
    def metadata(self):
        return {n: getattr(self, n) for n in Executable.FIELDS}
    
    @cached_property
    def ctime(self):
        return datetime.fromtimestamp(self.stat().st_ctime)
    
    @property
    def destination(self):
        try:
            return self._destination
        except AttributeError:
            try:
                return self._dataset.files.joinpath(self.hash)
            except AttributeError:
                pass
    
    @cached_property
    def filetype(self):
        try:
            return from_file(str(self))
        except OSError:
            return
    
    @cached_property
    def format(self):
        best_fmt, l = None, 0
        for ftype, fmt in Executable.SIGNATURES.items():
            if len(ftype) > l and is_filetype(str(self), ftype):
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def hash(self):
        return getattr(hashlib, Executable.HASH + "_file")(str(self))
    
    @cached_property
    def mtime(self):
        return datetime.fromtimestamp(self.stat().st_mtime)
    
    @cached_property
    def realpath(self):
        return str(self)
    
    @cached_property
    def size(self):
        return super(Executable, self).size

