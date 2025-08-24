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
from ...helpers import *

lazy_load_module("bintropy")
lazy_load_module("exeplot")


__all__ = ["is_exe", "is_elf", "is_macho", "is_pe", "Executable"] + _alter + _features

_N_SECTIONS = re.compile(r", \d+ sections$")
_METHOD = """def {name}(self, *args, **kwargs):
    from exeplot import {name}
    kwargs['img_name'] = self._filename(**kwargs)
    return {name}(*args, **kwargs)
"""
is_exe   = lambda e: Executable(e).format is not None
is_elf   = lambda e: is_exe(e) and Executable(e).group == "ELF"
is_macho = lambda e: is_exe(e) and Executable(e).group == "Mach-O"
is_pe    = lambda e: is_exe(e) and Executable(e).group == "PE"


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
    class Plot:
        def __init__(self, outer):
            from exeplot import __all__ as _viz_funcs
            self._outer = outer
            for f in _viz_funcs:
                exec(_METHOD.format(name=f))
                setattr(Executable.Plot, f, locals()[f])
        
        def _filename(self, prefix="", suffix="", no_ext=True, **kwargs):
            fn = (prefix or "") + Path(self._outer.realpath).basename + (suffix or "")
            if hasattr(self._outer, "_dataset"):
                fn = Path(self._outer._dataset.basename, "samples", fn)
            path = figure_path(fn, **kwargs)
            return str(path.dirname.joinpath(path.stem) if no_ext else path)
    
    def __init__(self, *args, **kwargs):
        self.__logger = kwargs.get('logger', null_logger)
        self.plot = self.Plot(self)
    
    def __new__(cls, *parts, **kwargs):
        h, ds1, ds2 = kwargs.pop('hash', None), kwargs.pop('dataset', None), kwargs.pop('dataset2', None)
        fields, kwargs['expand'], e = ["hash", "label"] + EXE_METADATA, True, parts[0] if len(parts) == 1 else None
        def _setattrs(exe, h):
            exe._dataset = ds1
            exe._destination = ds1.path.joinpath("files", h or exe.hash)
            # AttributeError: this occurs when the dataset is still empty, therefore holding no 'hash' column
            # IndexError:     this occurs when the executable did not exist in the dataset yet
            with suppress(AttributeError, IndexError):
                d, meta_fields = ds1._data[ds1._data.hash == h].iloc[0].to_dict(), EXE_METADATA + ["hash", "label"]
                for a in meta_fields:
                    # in some cases, we may instantiate an executable coming from a dataset and pointing on an altered
                    #  sample that has been corrupted ; then it is necessary to recompute the filetype (aka signature)
                    #  and the format attributes
                    if a in ["format", "signature"]:
                        continue
                    setattr(exe, a, d[a])
                f = {a: v for a, v in d.items() if a not in fields and str(v) != "nan"}
                # if features could be retrieved, set the 'data' attribute (i.e. if already computed as it is part of a
                #  FilelessDataset)
                if len(f) > 0:
                    setattr(exe, "data", f)
            if ds2 is not None and ds2._files:
                exe._destination = ds2.files.joinpath(h)
        # case (1) orphan executable (no dataset)
        if len(parts) > 0 and all(x is None for x in [h, ds1, ds2]):
            label = kwargs.pop('label', NOT_LABELLED)
            if len(parts) == 1:
                # if reinstantiating an Executable instance, simply immediately return it
                if isinstance(e, Executable):
                    return e
                elif isinstance(e, Path):
                    self = super(Executable, cls).__new__(cls, e, **kwargs)
                    self.label = label
                    self._case = "orphan executable"
                    return self
                parts = str(e).split(os.sep)
                if parts[0] == "":
                    parts[0] = os.sep
            # if the target executable has a path that may indicate it is part of a dataset, automatically set it
            if len(parts) >= 2 and parts[-2] == "files":
                from ..dataset import Dataset
                exe = Executable(hash=parts[-1], dataset=Dataset(Path(*parts[:-2])))
                if not exe.exists():
                    self._logger(**kwargs).warning("This executable does not exist ; this is because it comes from a"
                                                   " fileless dataset, meaning that computed features won't be computed"
                                                   " but will be retrieved from this dataset !")
                return exe
            self = super(Executable, cls).__new__(cls, *parts, **kwargs)
            self.label = label
            self._case = "orphan executable"
            return self
        # case (2) orphan excutable (dataset to be bound with)
        elif len(parts) > 0 and ds1 is not None and all(x is None for x in [h, ds2]) and not hasattr(e, "_fields"):
            self = super(Executable, cls).__new__(cls, *parts, **kwargs)
            self._case = f"bound to {ds1.basename} (no field)"
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
            self._case = f"bound to {ds1.basename} (fields from dictionary)"
            _setattrs(self, e.hash)
            for f in fields:
                #if f in ["format", "signature"]:
                #    continue
                setattr(self, f, getattr(e, f))
            return self
        # case (4) dataset-bound by hash (when ds2 is None)
        # case (5) dataset-bound, with a new destination dataset (when ds2 is not None)
        elif len(parts) == 0 and all(x is not None for x in [h, ds1]):
            exe = ds1.path.joinpath("files", h)
            self = super(Executable, cls).__new__(cls, str(exe), **kwargs)
            self._case = f"bound to {ds1.basename} ({'by hash' if ds2 is None else 'copy to ' + ds2.basename})"
            _setattrs(self, h)
            return self
        raise ValueError("Unsupported combination of arguments (see the docstring of the Executable class for more "
                         "information)")
    
    def _logger(self, **kwargs):
        return kwargs.get('logger', self.__logger)
    
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
        self._reset()
    
    @cached_result
    def block_entropy(self, blocksize=0, ignore_half_block_zeros=False, ignore_half_block_same_byte=True):
        return bintropy.entropy(self.read_bytes(), blocksize, ignore_half_block_zeros, ignore_half_block_same_byte)
    
    def compare(self, file2):
        from ...helpers.files import compare_files
        return compare_files(self, file2)
    
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
    
    def diff(self, file2, legend1=None, legend2=None, n=0, **kwargs):
        """ Generates a text-based difference between two PE files. 
        
        :param file1:   first file's name
        :param file2:   second file's name
        :param legend1: first file's alias (file1 if None)
        :param legend2: second file's alias (file2 if None)
        :param n:       amount of carriage returns between the sequences
        :return:        difference between the files, in text format
        """
        #FIXME: this only applies to PE ; need to find another way for ELF and Mach-O
        from difflib import unified_diff as udiff
        from pefile import PE
        self._logger(**kwargs).debug("dumping files info...")
        dump1, dump2 = PE(self).dump_info(), PE(file2).dump_info()
        return "\n".join(udiff(dump1.split('\n'), dump2.split('\n'), legend1 or str(self), legend2 or str(file2), n=n))
    
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
                raise ValueError(f"Modifier '{modifier}' does not exist")
        self.parsed.modify(modifier, **kwargs)
    
    def ngrams_counts(self, n=1):
        return exeplot.utils.ngrams_counts(self, n=n)
    
    def ngrams_distribution(self, n=1, n_most_common=None, n_exclude_top=0, exclude=None):
        return exeplot.utils.ngrams_distribution(self, n=n, n_most_common=n_most_common, n_exclude_top=n_exclude_top,
                                                 exclude=exclude)
    
    def objdump(self, n=0, executable_only=False):
        from subprocess import check_output
        output, result, l = check_output(["objdump", ["-D", "-d"][executable_only], str(self)]), bytearray(b""), 0
        for line in output.splitlines():
            tokens = line.decode().split("\t")
            if len(tokens) != 3:
                continue
            hb = tokens[1].strip().replace(" ", "")
            if re.match(r"([0-9a-f]{2})+", hb):
                b = bytes.fromhex(hb)
                l += len(b)
                if 0 < n < l:
                    return bytes(result + b[:l % n])
                result += b
        return bytes(result)
    
    def parse(self, parser=None, reset=True):
        parser = parser or config[f'{self.shortgroup}_parser']
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
    
    def scan(self, executables=(), **kwargs):
        l, verb = self._logger(**kwargs), kwargs.get('verbose', False) and len(executables) == 0
        @json_cache("vt", self.hash, kwargs.get('force', False))
        def _scan():
            from vt import Client
            k = config['vt_api_key']
            if k == "":
                l.warn("'vt_api_key' shall be defined")
                return
            try:
                with Client(k) as c:
                    l.info(f"Scanning '{self.filename}' ({self.hash}) on VirusTotal, this may take a while...")
                    with self.open('rb') as f:
                        analysis = c.scan_file(f, wait_for_completion=True)
                d = analysis.to_dict()
                d['filename'] = self.filename
                d['hash'] = self.hash
                return d
            except Exception as e:
                l.error(e)
                return
        data = _scan() or {}
        if not kwargs.get('show', True):
            return data
        sdata = [Executable(e).scan(show=False, **kwargs) for e in executables]
        _bad, _fail = ["malicious", "suspicious"], ["confirmed-timeout", "failure", "timeout", "type-unsupported"]
        sd = lambda d, k1, k2: d['attributes']['results'].get(k1, {}).get(k2, "")
        fmt = lambda r, c: colored(r or ("⚠" if c in _fail else ""),
                                   {'malicious': "red", 'suspicious': "orange"}.get(c, "grey"))
        st = lambda d: (d['filename'], d['hash'], sum(v for k, v in d['attributes']['stats'].items() if k in _bad), \
                        sum(v for k, v in d['attributes']['stats'].items() if k not in _fail))
        try:
            detections = [([d['engine_name'], d['engine_version'], d['engine_update'], d['category'], d['method'],
                            d['result'] or ""] if verb else [d['engine_name'], fmt(d['result'], d['category'])] + \
                       ([fmt(sd(x, d['engine_name'], 'result'), sd(x, d['engine_name'], 'category')) for x in sdata])) \
                        for d in sorted(data['attributes']['results'].values(), key=lambda x: x['engine_name'].lower())]
            stats = [st(data)] + [st(x) for x in sdata]
        except KeyError:
            l.error("Could not parse results")
            return
        head = [["Engine"] + [f"{f} ({m}/{t})\n{h}" for f, h, m, t in stats],
                ["Engine", "Version", "Last Update", "Category", "Method", "Result"]][verb]
        render(*[Section("Scan Results"), Table(detections, column_headers=head)])
    
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
                 f"**Block entropy**: {self.block_entropy(256)}"]
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
                sbe = sec.block_entropy(256)[1]
                row.append("-" if sec.size == 0 or sbe is None else f"{sbe:.5f}")
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
    def cfg(self):
        from .cfg import CFG
        return CFG(self)
    
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
        raise ValueError(f"Could not compute destination path for '{self}' (no destination specified or dataset bound)")
    
    @cached_property
    def entropy(self):
        return bintropy.entropy(self.read_bytes())
    
    @cached_property
    def features(self):
        try:
            return self._dataset._features
        except AttributeError:
            if self.format is not None:
                Features()  # lazily populate Features.registry at first instantiation
                return {n: f.description for n, f in Features.registry[self.format].items() if f.keep}
    
    @property
    def filetype(self):
        return self.signature
    
    @cached_property
    def format(self):
        best_fmt, l = None, 0
        if self.filetype is None:
            raise OSError(f"'{self}' does not exist ({self._case})")
        # NB: the best signature matched is the longest
        for ftype, fmt in SIGNATURES.items():
            if len(ftype) > l and self.filetype is not None and re.search(ftype, self.filetype) is not None:
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def fuzzy_hash(self):
        return globals()[config['fuzzy_hash_algorithm'].replace("-", "_")](self)
    
    @cached_property
    def group(self):
        return get_format_group(self.format)
    
    @cached_property
    def hash(self):
        return getattr(hashlib, config['hash_algorithm'] + "_file")(str(self))
    
    @property
    def metadata(self):
        return {n: getattr(self, n) for n in EXE_METADATA}
    
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
    
    @cached_property
    def signature(self):
        from magic import from_file
        try:
            return _N_SECTIONS.sub("", from_file(str(self)))
        except OSError:
            return
    
    @cached_property
    def size(self):
        return super(Executable, self).size

