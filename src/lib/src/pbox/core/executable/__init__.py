# -*- coding: UTF-8 -*-
from contextlib import suppress
from datetime import datetime
from functools import cached_property
from tinyscript import hashlib, re, shutil
from tinyscript.helpers import lazy_load_module, Path, TempPath

from .alterations import *
from .alterations import __all__ as _alter
from .features import *
from .features import __all__ as _features
from .visualization import *
from ...helpers.formats import get_format_group


__all__ = ["is_exe", "Executable"] + _alter + _features


is_exe = lambda e: Executable(e).format is not None


class Executable(Path):
    """ Executable abstraction.
    
    Can be initialized in different ways:
    (1) orphan executable (not coming from a dataset)
    (2) dataset-bound (data is used to populate attributes based on the ; this does not require the file)
    (3) dataset-bound, with a new destination dataset
    """
    FIELDS = ["realpath", "format", "signature", "size", "ctime", "mtime"]
    HASH = config['hash_algorithm']  # possible values: hashlib.algorithms_available
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

    def __new__(cls, *parts, **kwargs):
        data, fields = None, ["hash", "label"] + Executable.FIELDS
        ds1, ds2 = kwargs.pop('dataset', None), kwargs.pop('dataset2', None)
        if len(parts) == 1:
            e = parts[0]
            # if reinstantiating an Executable instance, simply immediately return it
            if isinstance(e, Executable) and not kwargs.get('force', False):
                return e
            # case (1) occurs when using pandas.DataFrame.itertuples() ; this produces an instance of
            #  pandas.core.frame.Pandas (having the '_fields' attribute), hence causing an orphan executable
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
            # AttributeError: this occurs when the dataset is still empty, therefore holding no 'hash' column
            # IndexError:     this occurs when the executable did not exist in the dataset yet
            with suppress(AttributeError, IndexError):
                d, meta_fields = ds1._data[ds1._data.hash == h].iloc[0].to_dict(), Executable.FIELDS + ["hash", "label"]
                for a in meta_fields:
                    setattr(self, a, d[a])
                f = {a: v for a, v in d.items() if a not in fields if str(v) != "nan"}
                # if features could be retrieved, set the 'data' attribute (i.e. if already computed as it is part of a
                #  FilelessDataset)
                if len(f) > 0:
                    setattr(self, "data", f)
            # case (3)
            if ds2 is not None and ds2._files:
                self._destination = ds2.files.joinpath(h)
        self.label = kwargs.pop('label', getattr(self, "label", NOT_LABELLED))
        return self
    
    def copy(self, extension=False, overwrite=False):
        dest = Executable(str(self.destination) + ["", self.extension.lower()][extension])
        if str(self) != dest:
            if overwrite and dest.exists():
                dest.remove()
            try:  # copy file with its attributes and metadata
                shutil.copy2(str(self), str(dest))
                dest.chmod(0o400)
                return dest
            except:
                pass
        # return None to indicate that the copy failed (i.e. when the current instance is already the destination path)
    
    def update(self):
        # ensure filetype and format will be recomputed at their next invocation
        self.__dict__.pop('filetype', None)
        self.__dict__.pop('format', None)
        self.__dict__.pop('hash', None)
    
    @staticmethod
    def is_valid(path):
        return Executable(path).format is not None
    
    @property
    def metadata(self):
        return {n: getattr(self, n) for n in Executable.FIELDS}
    
    @cached_property
    def ctime(self):
        return datetime.fromtimestamp(self.stat().st_ctime)
    
    @cached_property
    def data(self):
        return Features(self)
    
    @property
    def destination(self):
        if hasattr(self, "_destination"):
            return self._destination
        if hasattr(self, "_dataset") and self.hash is not None:
            return (self._dataset.files if self._dataset._files else TempPath()).joinpath(self.hash)
        raise ValueError("Could not compute destination path for '%s'" % self)
    
    @property
    def features(self):
        Features(None)  # lazily populate Features.registry at first instantiation
        if self.format is not None:
            return {n: f.description for n, f in Features.registry[self.format].items()}
    
    @cached_property
    def filetype(self):
        from magic import from_file
        try:
            return from_file(str(self))
        except OSError:
            return
    
    @cached_property
    def format(self):
        best_fmt, l = None, 0
        for ftype, fmt in Executable.SIGNATURES.items():
            if len(ftype) > l and re.search(ftype, self.filetype) is not None:
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def group(self):
        return get_format_group(self.format)
    
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
    def signature(self):
        return self.filetype
    
    @cached_property
    def size(self):
        return super(Executable, self).size
    
    @staticmethod
    def diff_plot(file1, file2, img_name=None, img_format="png", legend1="", legend2="", dpi=400, title=None, **kwargs):
        return binary_diff_plot(file1, file2, img_name, img_format, legend1, legend2, dpi, title, **kwargs)
    
    @staticmethod
    def diff_text(file1, file2, legend1=None, legend2=None, n=0, **kwargs):
        return binary_diff_text(file1, file2, legend1, legend2, n, **kwargs)

