# -*- coding: UTF-8 -*-
from contextlib import suppress
from datetime import datetime
from tinyscript import hashlib, os, re, shutil
from tinyscript.helpers import human_readable_size, Path, TempPath
from tinyscript.report import *

from .alterations import *
from .alterations import __all__ as _alter
from .features import *
from .features import __all__ as _features
from .parsers import get_parser
from .visualization import *
from .visualization import __all__ as _viz
from ...helpers import *

lazy_load_module("bintropy")


__all__ = ["is_exe", "Executable"] + _alter + _features + _viz

_N_SECTIONS = re.compile(r", \d+ sections$")

is_exe = lambda e: Executable(e).format is not None


class Executable(Path):
    """ Executable abstraction.
    
    Can be initialized in different ways:
    (1) orphan executable (not coming from a dataset)
         args: parts
        NB: mapped to dataset-bound if the given path ends by 'files/[hash]'
    (2) orphan executable (to be bound to a dataset)
         args:   parts (not a data row)
         kwargs: dataset
    (3) dataset-bound by data row (data is used to populate attributes ; this does not require the file)
         args:   data row
         kwargs: dataset
    (4) dataset-bound by hash (data is used to populate attributes ; this does not require the file)
         kwargs: hash, dataset
    (5) dataset-bound, with a new destination dataset
         kwargs: hash, dataset, dataset2
    """
    FIELDS = ["realpath", "format", "signature", "size", "ctime", "mtime"]
    
    def __new__(cls, *parts, **kwargs):
        h, ds1, ds2 = kwargs.pop('hash', None), kwargs.pop('dataset', None), kwargs.pop('dataset2', None)
        fields, kwargs['expand'], e = ["hash", "label"] + Executable.FIELDS, True, parts[0] if len(parts) == 1 else None
        def _setattrs(exe, hash):
            exe._dataset = ds1
            exe._destination = ds1.path.joinpath("files", hash or exe.hash)
            # AttributeError: this occurs when the dataset is still empty, therefore holding no 'hash' column
            # IndexError:     this occurs when the executable did not exist in the dataset yet
            with suppress(AttributeError, IndexError):
                d, meta_fields = ds1._data[ds1._data.hash == h].iloc[0].to_dict(), Executable.FIELDS + ["hash", "label"]
                for a in meta_fields:
                    # in some cases, we may instantiate an executable coming from a dataset and pointing on an altered
                    #  sample that has been corrupted ; then it is necessary to recompute the filetype (aka signature)
                    #  and the format attributes
                    if a in ["format", "signature"]:
                        continue
                    setattr(exe, a, d[a])
                f = {a: v for a, v in d.items() if a not in fields if str(v) != "nan"}
                # if features could be retrieved, set the 'data' attribute (i.e. if already computed as it is part of a
                #  FilelessDataset)
                if len(f) > 0:
                    setattr(exe, "data", f)
            if ds2 is not None and ds2._files:
                exe._destination = ds2.files.joinpath(h)
        # case (1) orphan excutable (no dataset)
        if len(parts) > 0 and all(x is None for x in [h, ds1, ds2]):
            label = kwargs.pop('label', NOT_LABELLED)
            if len(parts) == 1:
                # if reinstantiating an Executable instance, simply immediately return it
                if isinstance(e, Executable):
                    return e
                elif isinstance(e, Path):
                    self = super(Executable, cls).__new__(cls, e, **kwargs)
                    self.label = label
                    return self
                parts = str(e).split(os.sep)
                if parts[0] == "":
                    parts[0] = os.sep
            # if the target executable has a path that may indicate it is part of a dataset, automatically set it
            if len(parts) >= 2 and parts[-2] == "files":
                from ..dataset import Dataset
                return Executable(hash=parts[-1], dataset=Dataset(Path(*parts[:-2])))
            self = super(Executable, cls).__new__(cls, *parts, **kwargs)
            self.label = label
            return self
        # case (2) orphan excutable (dataset to be bound with)
        elif len(parts) > 0 and ds1 is not None and all(x is None for x in [h, ds2]) and not hasattr(e, "_fields"):
            self = super(Executable, cls).__new__(cls, *parts, **kwargs)
            _setattrs(self)
            return self
        # case (3) dataset-bound by data row
        elif len(parts) == 1 and ds1 is not None and all(x is None for x in [h, ds2]) and \
             all(hasattr(e, f) for f in fields) and hasattr(e, "_fields"):
            # occurs when using pandas.DataFrame.itertuples() ; this produces an instance of pandas.core.frame.Pandas
            #  (having the '_fields' attribute), hence causing an orphan executable
            try:
                dest = ds1.files.joinpath(e.hash)
            except AttributeError:  # 'NoneType|FilelessDataset' object has no attribute 'files'
                dest = e.realpath
            self = super(Executable, cls).__new__(cls, dest, **kwargs)
            _setattrs(self, e.hash)
            for f in fields:
                if f in ["format", "signature"]:
                    continue
                setattr(self, f, getattr(e, f))
            return self
        # case (4) dataset-bound by hash (when ds2 is None)
        # case (5) dataset-bound, with a new destination dataset (when ds2 is not None)
        elif len(parts) == 0 and all(x is not None for x in [h, ds1]):
            exe = ds1.path.joinpath("files", h)
            self = super(Executable, cls).__new__(cls, str(exe), **kwargs)
            _setattrs(self, h)
            return self
        raise ValueError("Unsupported combination of arguments (see the docstring of the Executable class for more "
                         "information)")
    
    def _reset(self):
        # ensure filetype and format will be recomputed at their next invocation (i.e. when a packer modifies the binary
        #  in a way that makes its format change)
        for attr in list(self.__dict__.keys()):
            if not isinstance(getattr(Executable, attr, None), cached_property):
                continue
            try:
                delattr(self, attr)
            except AttributeError:
                pass
    
    def alter(self, *alterations, **kwargs):
        Alterations(self, alterations or None)
    
    def block_entropy(self, blocksize=0, ignore_half_block_zeros=False):
        return bintropy.entropy(self.read_bytes(), blocksize, ignore_half_block_zeros)
    
    def copy(self, extension=False, overwrite=False):
        # NB: 'dest' is not instantiated with the Executable class as it does not exist yet
        dest = Path(str(self.destination) + ["", self.extension.lower()][extension])
        if str(self) != dest:
            if overwrite and dest.exists():
                dest.remove()
            try:  # copy file with its attributes and metadata
                shutil.copy2(str(self), str(dest))
                dest.chmod(0o400)
                return Executable(str(dest))
            except:
                pass
        # return None to indicate that the copy failed (i.e. when the current instance is already the destination path)
    
    def is_valid(self):
        return self.format is not None
    
    def modify(self, modifier, *args, **kwargs):
        if isinstance(modifier, str):
            from .modifiers import Modifiers
            found = False
            for m, f in Modifiers()[self.format].items():
                if m == modifier:
                    modifier = f(*args, **kwargs)
                    found = True
                    break
            if not found:
                raise ValueError("Modifier '%s' does not exist" % modifier)
        self.parsed.modify(modifier, **kwargs)
    
    def parse(self, parser=None, reset=True):
        parser = parser or config['%s_parser' % self.shortgroup]
        if reset:  # this forces recomputing the cached properties 'parsed', 'hash', 'size', ...
            self._reset()
        elif hasattr(self, "_parsed") and self._parsed.parser == parser:
            return self._parsed
        self._parsed = get_parser(parser, self.format)(str(self))
        self._parsed.parser = parser  # keeps the name of the parser used
        self._parsed.path = self      # sets a back reference
        if self.group == "PE":
            self._parsed.real_section_names  # trigger computation of real names
        return self._parsed
    
    def plot(self, prefix="", sublabel="size-ep-ent", **kwargs):
        fn = (prefix or "") + Path(self.realpath).basename
        if hasattr(self, "_dataset"):
            fn = Path(self._dataset.basename, "samples", fn)
        path = figure_path(fn, **kwargs)
        kwargs.get('logger', null_logger).info(f"Saving to {path}...")
        path = path.dirname.joinpath(path.stem)
        kw_plot = {k: kwargs.get(k, config[k]) for k in config._defaults['visualization'].keys()}
        kw_plot['img_format'] = kw_plot.pop('format', config['format'])
        from bintropy import plot
        plot(self, img_name=path, labels=[self.label], sublabel=sublabel, target=fn, **kw_plot)
    
    def show(self, base_info=True, sections=True, features=False, **kwargs):
        ds, r = hasattr(self, "_dataset"), []
        if base_info:
            l = [f"**Hash**:          {self.hash}",
                 f"**Filename**:      {Path(self.realpath).basename}",
                 f"**Format**:        {self.format}",
                 f"**Signature**:     {self.signature}",
                 f"**Entry point**:   0x{self.parsed.entrypoint:02X} ({self.parsed.entrypoint_section.name})",
                 f"**Size**:          {human_readable_size(self.size)}",
                 f"**Entropy**:       {self.entropy}",
                 f"**Block entropy**: {self.block_entropy_256B}"]
            if ds:
                l += [f"**Label**:         {self.label}"]
            r += [Section("Base Information"), List(l)]
        if sections:
            h, s = [], []
            for i, sec in enumerate(self.parsed):
                row = []
                if i == 0:
                    h.append("Name")
                rn = f" ({sec.real_name})" if hasattr(sec, "real_name") and sec.real_name != sec.name else ""
                row.append(f"{sec.name}{rn}")
                if i == 0:
                    h.append("Raw size")
                row.append(human_readable_size(sec.size))
                if hasattr(sec, "virtual_size"):
                    if i == 0:
                        h.append("Virtual size")
                    row.append(human_readable_size(sec.virtual_size))
                if i == 0:
                    h.extend(["Entropy", "Block entropy (256B)"])
                row.append(f"{sec.entropy:.5f}" if sec.size > 0 else "-")
                row.append("-" if sec.size == 0 or sec.block_entropy_256B is None else f"{sec.block_entropy_256B:.5f}")
                if i == 0:
                    h.append("Standard")
                row.append(f"{'NY'[sec.is_standard]}")
                s.append(row)
            r += [Section("Sections"), Table(s, column_headers=h)]
        if features:
            maxlen = max(map(len, self.data.keys()))
            f = [f"{'**'+n+'**:':<{maxlen+6}}{v}" for n, v in self.data.items()]
            r += [Section("Features"), List(f)]
        render(*r)
    
    @property
    def bin_label(self):
        return READABLE_LABELS(self.label, True)
    
    @cached_property
    def block_entropy_256B(self):
        return bintropy.entropy(self.read_bytes(), 256, True)[1]
    
    @cached_property
    def block_entropy_512B(self):
        return bintropy.entropy(self.read_bytes(), 512, True)[1]
    
    @cached_property
    def cfg(self):
        from .cfg import CFG
        cfg = CFG(self)
        # found_node_at_ep = cfg.model.get_any_node(cfg.model.project.entry) is not None
        # n_nodes = len(cfg.model.graph.nodes())
        cfg.compute()
        return cfg
    
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
    
    @cached_property
    def entropy(self):
        return bintropy.entropy(self.read_bytes())
    
    @cached_property
    def features(self):
        Features()  # lazily populate Features.registry at first instantiation
        if self.format is not None:
            return {n: f.description for n, f in Features.registry[self.format].items()}
    
    @cached_property
    def filetype(self):
        from magic import from_file
        try:
            return _N_SECTIONS.sub("", from_file(str(self)))
        except OSError:
            return
    
    @cached_property
    def format(self):
        best_fmt, l = None, 0
        # NB: the best signature matched is the longest
        for ftype, fmt in SIGNATURES.items():
            if len(ftype) > l and self.filetype is not None and re.search(ftype, self.filetype) is not None:
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def group(self):
        return get_format_group(self.format)
    
    @cached_property
    def hash(self):
        return getattr(hashlib, config['hash_algorithm'] + "_file")(str(self))
    
    @property
    def metadata(self):
        return {n: getattr(self, n) for n in Executable.FIELDS}
    
    @cached_property
    def mtime(self):
        return datetime.fromtimestamp(self.stat().st_mtime)
    
    @cached_property
    def parsed(self):
        if not hasattr(self, "_parsed"):
            self.parse()  # this will use the default parser configured in ~/.packing-box.conf
        return self._parsed
    
    @property
    def rawdata(self):
        # NB: this property is for debugging, it never gets called within pbox
        from .extractors import Extractors
        return Extractors(self)
    
    @cached_property
    def realpath(self):
        return str(self)
    
    @cached_property
    def shortgroup(self):
        return get_format_group(self.format, True)
    
    @property
    def signature(self):
        return self.filetype
    
    @cached_property
    def size(self):
        return super(Executable, self).size
    
    @staticmethod
    def diff_plot(file1, file2, img_name=None, img_format="png", legend1="", legend2="", dpi=400, title=None, **kwargs):
        return binary_diff_plot(Executable(file1), Executable(file2), img_name, img_format, legend1, legend2, dpi,
                                title, **kwargs)
    
    @staticmethod
    def diff_text(file1, file2, legend1=None, legend2=None, n=0, **kwargs):
        return binary_diff_text(Executable(file1), Executable(file2), legend1, legend2, n, **kwargs)

