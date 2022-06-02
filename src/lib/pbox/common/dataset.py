# -*- coding: UTF-8 -*-
import pandas as pd
import re
from datetime import datetime, timedelta
from textwrap import wrap
from tinyscript import b, colored, hashlib, json, logging, random, subprocess, time, ts
from tinyscript.report import *
from tqdm import tqdm

from .config import *
from .executable import *
from .utils import *
from ..items import *


__all__ = ["Dataset", "PACKING_BOX_SOURCES"]


BACKUP_COPIES = 3
PACKING_BOX_SOURCES = {
    'ELF': ["/sbin", "/usr/bin"],
    'PE':  ["/root/.wine/drive_c/windows", "/root/.wine32/drive_c/windows"],
}


class Dataset:
    """ Folder structure:
    
    [name]
      +-- files
      |     +-- {executables, renamed to their SHA256 hashes}
      +-- data.csv          # metadata and labels of the executable
      +-- metadata.json     # simple statistics about the dataset
    """
    @logging.bindLogger
    def __init__(self, name="dataset", source_dir=None, load=True, **kw):
        if not re.match(NAMING_CONVENTION, name.basename if isinstance(name, ts.Path) else str(name)):
            raise ValueError("Bad input name")
        self._files = getattr(self.__class__, "_files", True)
        self.path = ts.Path(config['datasets'].joinpath(name), create=load).absolute()
        self.sources = source_dir or PACKING_BOX_SOURCES
        if isinstance(self.sources, list):
            self.sources = {'All': [str(x) for x in self.sources]}
        for _, sources in self.sources.items():
            for source in sources[:]:
                s = ts.Path(source, expand=True)
                if not s.exists() or not s.is_dir():
                    sources.remove(source)
        if load:
            self._load()
        self.formats = getattr(self, "_metadata", {}).get('formats', [])
        self.__change = False
        self.__limit = 20
        self.__per_format = False
    
    def __delitem__(self, executable):
        """ Remove an executable (by its real name or hash) from the dataset. """
        self.__change, df = True, self._data
        h = executable
        # first, ensure we handle the hash name (not the real one)
        try:
            h = df.loc[df['realpath'] == h, 'hash'].iloc[0]
        except:
            if isinstance(h, Executable):
                h = h.hash
        if len(df) > 0:
            self.logger.debug("removing %s..." % h)
            self._data = df[df.hash != h]
        if self._files:
            self.files.joinpath(h).remove(error=False)
            try:
                ext = ts.Path(df.loc[df['hash'] == h, 'realpath'].iloc[0]).extension
                self.files.joinpath(h + ext).remove(error=False)
            except:
                pass
    
    def __eq__(self, dataset):
        """ Custom equality function. """
        ds = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        return self._metadata == ds._metadata and self._data.equals(ds._data) and \
               (not self._files or list(self.files.listdir()) == list(ds.files.listdir()))
    
    def __getitem__(self, hash):
        """ Get a data row related to the given hash from the dataset. """
        try:
            h, with_headers = hash
        except ValueError:
            h, with_headers = hash, False
        if len(self._data) == 0:
            raise KeyError(h)
        try:
            row = self._data[self._data.hash == h].iloc[0]
            return row.to_dict() if with_headers else row.to_list()
        except IndexError:
            raise KeyError(h)
    
    def __hash__(self):
        """ Custom object hashing function. """
        return int.from_bytes(hashlib.md5(b(self.name)).digest(), "little")
    
    def __len__(self):
        """ Get dataset's length. """
        return len(self._data)
    
    def __repr__(self):
        """ Custom string representation. """
        return "<%s dataset at 0x%x>" % (self.name, id(self))
    
    def __setattr__(self, name, value):
        # auto-expand the formats attribute into a private one
        if name == "formats":
            self._formats_exp = expand_formats(*value)
        super(Dataset, self).__setattr__(name, value)
    
    def __setitem__(self, executable, label):
        """ Add an executable based on its real name to the dataset.
        
        :param executable: either the path to the executable or its Executable instance
        :param label:      either the text label of the given executable or its dictionary of data
        """
        try:
            label, update = label
        except (TypeError, ValueError):  # TypeError occurs when label is None
            label, update = label, False
        self.__change, l = True, self.logger
        df, e = self._data, executable
        e = e if isinstance(e, Executable) else Executable(e, dataset=self)
        # consider the case when 'label' is a dictionary with the executable's attribute, i.e. from another dataset
        if isinstance(label, dict):
            d = label
            # get metadata values from the input dictionary
            for k, v in d.items():
                if k in Executable.FIELDS:
                    setattr(e, k, v)
            # then ensure we compute the remaining values
            for k in ['hash'] + Executable.FIELDS:
                if k not in d:
                    d[k] = getattr(e, k)
        # then consider the case when 'label' is really the label value
        else:
            d = e.metadata
            d['hash'] = e.hash
            d['label'] = e.label = label
        if self._files:
            e.copy()
        self._metadata.setdefault('formats', [])
        if e.format not in self._metadata['formats']:
            self._metadata['formats'].append(e.format)
        if len(df) > 0 and e.hash in df.hash.values:
            row = df.loc[df['hash'] == e.hash]
            lbl = row['label'].iloc[0]
            # consider updating when:
            #  (a) hash already exists but is not packed => can pack it
            #  (b) new fields are added, e.g. when converting to FilelessDataset (features are computed)
            if str(lbl) == "nan" or update:
                l.debug("updating %s..." % e.hash)
                for n, v in d.items():
                    df.loc[df['hash'] == e.hash, n] = v
            else:
                l.debug("discarding %s%s..." % (e.hash, ["", " (already in dataset)"][lbl == d['label']]))
        else:
            l.debug("adding %s..." % e.hash)
            self._data = df.append(d, ignore_index=True)
    
    def __str__(self):
        """ Custom object's string. """
        return self.name
    
    def _copy(self, path):
        """ Copy the current dataset to a given destination. """
        self.path.copy(path)
    
    def _filter(self, query=None, **kw):
        """ Yield executables' hashes from the dataset using Pandas' query language. """
        i = -1
        try:
            for i, row in enumerate(self._data.query(query or "tuple()").itertuples()):
                yield row
            if i == -1:
                self.logger.warning("No data selected")
        except (AttributeError, KeyError) as e:
            self.logger.error("Invalid query syntax ; %s" % e)
        except SyntaxError:
            self.logger.error("Invalid query syntax ; please checkout Pandas' documentation for more information")
        except pd.core.computation.ops.UndefinedVariableError as e:
            self.logger.error(e)
            self.logger.info("Possible values:\n%s" % "".join("- %s\n" % n for n in self._data.columns))
    
    def _load(self):
        """ Load dataset's associated files or create them. """
        self.logger.debug("loading dataset '%s'..." % self.basename)
        if self._files:
            self.files.mkdir(exist_ok=True)
        for n in ["data", "metadata"] + [["features"], []][self._files]:
            p = self.path.joinpath(n + [".json", ".csv"][n == "data"])
            if n == "data":
                try:
                    self._data = pd.read_csv(str(p), sep=";", parse_dates=['ctime', 'mtime'])
                except (OSError, KeyError):
                    self._data = pd.DataFrame()
            elif p.exists():
                with p.open() as f:
                    setattr(self, "_" + n, json.load(f))
            else:
                setattr(self, "_" + n, {})
    
    def _remove(self):
        """ Remove the current dataset. """
        self.logger.debug("removing dataset '%s'..." % self.basename)
        self.path.remove(error=False)
    
    def _save(self):
        """ Save dataset's state to JSON files. """
        if not self.__change:
            return
        if len(self) == 0 and not Dataset.check(self.basename):
            self._remove()
            return
        self.logger.debug("saving dataset '%s'..." % self.basename)
        self._metadata['formats'] = sorted(collapse_formats(*self._metadata['formats']))
        try:
            self._metadata['counts'] = self._data.label.value_counts().to_dict()
        except AttributeError:
            self.logger.warning("No label found")
            self._remove()
            return
        self._metadata['executables'] = len(self)
        for n in ["data", "metadata"] + [["features"], []][self._files]:
            if n == "data":
                self._data = self._data.sort_values("hash")
                fields = ["hash"] + Executable.FIELDS + ["label"]
                fnames = [h for h in self._data.columns if h not in fields + ["Index"]]
                c = fields[:-1] + fnames + [fields[-1]]
                self._data.to_csv(str(self.path.joinpath("data.csv")), sep=";", columns=c, index=False, header=True)
            else:
                with self.path.joinpath(n + ".json").open('w+') as f:
                    json.dump(getattr(self, "_" + n), f, indent=2)
        self.__change = False
    
    def _walk(self, walk_all=False, sources=None, silent=False):
        """ Walk the sources for random in-scope executables. """
        [self.logger.info, self.logger.debug][silent]("Searching for executables...")
        m, candidates, packers = 0, [], [p.name for p in Packer.registry]
        for cat, srcs in (sources or self.sources).items():
            if all(c not in expand_formats(cat) for c in self._formats_exp):
                continue
            for src in srcs:
                for exe in ts.Path(src, expand=True).walk(filter_func=lambda x: x.is_file(), sort=False):
                    exe = Executable(exe, dataset=self)
                    if exe.format not in self._formats_exp or exe.stem in packers:
                        continue  # ignore unrelated files and packers themselves
                    if walk_all:
                        yield exe
                    else:
                        candidates.append(exe)
        if len(candidates) > 0 and not walk_all:
            random.shuffle(candidates)
            for c in candidates:
                yield c
    
    def edit(self, **kw):
        """ Edit the data CSV file. """
        self.logger.debug("editing dataset's data.csv...")
        edit_file(self.path.joinpath("data.csv").absolute(), logger=self.logger)
    
    def exists(self):
        """ Dummy exists method. """
        return self.path.exists()
    
    def export(self, destination="export", n=0, **kw):
        """ Export packed executables to the given destination folder. """
        self.logger.info("Exporting packed executables of %s to '%s'..." % (self.basename, destination))
        l, tmp = [e for e in self if e.label is not None], []
        n = min(n, len(l))
        if n > 0:
            random.shuffle(l)
        pbar = tqdm(total=n or len(l), unit="packed executable")
        for i, exe in enumerate(l):
            if i >= n > 0:
                break
            fn = "%s_%s" % (exe.label, ts.Path(exe.realpath).filename)
            if fn in tmp:
                self.logger.warning("duplicate '%s'" % fn)
                n += 1
                continue
            exe.destination.copy(ts.Path(destination, create=True).joinpath(fn))
            tmp.append(fn)
            pbar.update()
        pbar.close()
    
    @backup
    def fix(self, **kw):
        """ Make dataset's structure and files match. """
        self.logger.debug("dropping duplicates...")
        self._data = self._data.drop_duplicates()
        for exe in self.files.listdir(ts.is_executable):
            h = exe.basename
            if h not in self._data.hash.values:
                del self[h]
        for exe in self:
            h = exe.hash
            if not self.files.joinpath(h).exists():
                del self[h]
        self._save()
    
    def is_empty(self):
        """ Check if this Dataset instance has a valid structure. """
        return len(self) == 0
    
    def is_valid(self):
        """ Check if this Dataset instance has a valid structure. """
        return self.__class__.check(self.path)
    
    def list(self, show_all=False, hide_files=False, raw=False, **kw):
        """ List all the datasets from the given path. """
        self.logger.debug("summarizing datasets from %s..." % config['datasets'])
        d = Dataset.summarize(str(config['datasets']), show_all, hide_files)
        if len(d) > 0:
            r = mdv.main(Report(*d).md())
            print(ts.ansi_seq_strip(r) if raw else r)
        else:
            self.logger.warning("No dataset found in workspace (%s)" % config['datasets'])
    
    @backup
    def make(self, n=0, formats=["All"], balance=False, packer=None, pack_all=False, **kw):
        """ Make n new samples in the current dataset among the given binary formats, balanced or not according to
             the number of distinct packers. """
        l, self.formats = self.logger, formats  # this triggers creating self._formats_exp
        # select enabled and non-failing packers among the input list
        packers = [p for p in (packer or Packer.registry) if p in Packer.registry and \
                                                             p.check(*self._formats_exp, silent=False)]
        if len(packers) == 0:
            l.critical("No valid packer selected")
            return
        # then restrict dataset's formats to these of the selected packers
        pformats = aggregate_formats(*[p.formats for p in packers])
        self.formats = collapse_formats(*[f for f in expand_formats(*formats) if f in pformats])
        sources = []
        for fmt, src in self.sources.items():
            if all(c not in expand_formats(fmt) for f in self._formats_exp):
                continue
            sources.extend(src)
        l.info("Source directories: %s" % ",".join(map(str, set(sources))))
        l.info("Considered formats: %s" % ",".join(self.formats))  # this updates self._formats_exp
        l.info("Selected packers:   %s" % ",".join(["All"] if packer is None else \
                                                   [p.__class__.__name__ for p in packer]))
        self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + sources)))
        if n == 0:
            n = len(list(self._walk(n <= 0, silent=True)))
        # get executables to be randomly packed or not
        CBAD, CGOOD = n // 3, n // 3
        i, cbad, cgood, pbar = 0, {p: CBAD for p in packer}, {p: CGOOD for p in packer}, None
        for exe in self._walk(n <= 0):
            label = short_label = None
            to_be_packed = pack_all or random.randint(0, len(packers) if balance else 1)
            # check 1: are there already samples enough?
            if i >= n > 0:
                break
            # check 2: are there working packers remaining?
            if len(packers) == 0:
                l.critical("No packer left")
                return
            # check 3: is the selected Executable supported by any of the remaining packers?
            if all(not p._check(exe, silent=True) for p in packers):
                l.debug("unsupported file (%s)" % exe)
                continue
            # check 4.a: was this executable already included in the dataset?
            if len(self._data) > 0 and not to_be_packed and exe.hash in self._data.hash.values:
                l.debug("already in the dataset (%s)" % exe)
                continue
            l.debug("handling %s..." % exe)
            # set the progress bar now to not overlap with self._walk's logging
            if pbar is None:
                pbar = tqdm(total=n, unit="executable")
            if to_be_packed:
                if len(packers) > 1:
                    random.shuffle(packers)
                destination = exe.copy(extension=True)
                if destination is None:  # occurs when the copy failed
                    continue
                old_h = destination.absolute()
                for p in packers[:]:
                    exe.hash, label = p.pack(str(old_h), include_hash=True)
                    if exe.hash is None:
                        continue  # means that this kind of executable is not supported by this packer
                    # check 4.b: was this packed executable already included in the dataset?
                    if len(self._data) > 0 and exe.hash in self._data.hash.values:
                        l.debug("already packed before (%s)" % exe)
                        continue
                    if not label or p._bad:
                        # if we reached the limit of GOOD packing occurrences, then we consider the packer as GOOD again
                        if cgood[p] <= 0:
                            p._bad, cbad[p] = False, n // 3
                        # but if BAD, we reset the GOOD counter and eventually disable it
                        if p._bad:
                            cbad[p] -= 1      # update BAD counter
                            cgood[p] = CGOOD  # reset GOOD counter ; if still in BAD state, then we need 'cgood'
                                              #  successful packings to return to the GOOD state
                            if cbad[p] <= 0:  # BAD counter exhausted => disable the packer
                                l.warning("Disabling %s..." % p.__class__.__name__)
                                packers.remove(p)
                        # if GOOD and label is None
                        else:
                            cgood[p] -= 1   # update GOOD counter
                            cbad[p] = CBAD  # reset BAD counter
                        label = None
                        continue
                    else:  # consider short label (e.g. "midgetpack", not "midgetpack[<password>]")
                        short_label = label.split("[")[0]
                    break
                # ensure we did not left the executable name with its hash AND extension behind
                try:
                    old_h.rename(exe.destination)
                except FileNotFoundError:
                    pass
            if not pack_all or (pack_all and short_label is not None):
                self[exe] = short_label
            if label is not None or not pack_all:
                i += 1
                pbar.update()
        if pbar:
            pbar.close()
        if i < n - 1:
            l.warning("Found too few candidate executables")
        else:
            ls = len(self)
            if ls > 0:
                p = sorted(list(set([lb for lb in self._data.label.values if isinstance(lb, str)])))
                l.info("Used packers: %s" % ", ".join(p))
                l.info("#Executables: %d" % ls)
        # finally, save dataset's metadata and labels to JSON files
        self._save()
    
    def purge(self, backup=False, **kw):
        """ Truncate and recreate a blank dataset. """
        self.logger.debug("purging %s%s..." % (self.path, ["", "'s backups"][backup]))
        if not backup:
            self._remove()
        # also recursively purge the backups
        try:
            self.backup.purge()
        except AttributeError:
            pass
    
    @backup
    def remove(self, query=None, **kw):
        """ Remove executables from the dataset given multiple criteria. """
        self.logger.debug("removing files from %s based on query '%s'..." % (self.basename, query))
        for e in self._filter(query, **kw):
            del self[e.hash]
        self._save()
    
    def rename(self, name2=None, **kw):
        """ Rename the current dataset. """
        l, path2 = self.logger, config['datasets'].joinpath(name2)
        if not path2.exists():
            l.debug("renaming %s (and backups) to %s..." % (self.basename, name2))
            tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
            self.path.rename(path2)
            self.path = path2
            tmp.rename(ts.TempPath(".dataset-backup", hex(hash(self))[2:]))
        else:
            l.warning("%s already exists" % name2)
    
    def revert(self, **kw):
        """ Revert to the latest version of the dataset (if a backup copy exists in /tmp). """
        b, l = self.backup, self.logger
        if b is None:
            l.warning("No backup found ; could not revert")
        else:
            l.debug("reverting %s to previous backup..." % self.basename)
            self._remove()
            b._copy(self.path)
            b._remove()
            self._save()
    
    def select(self, name2=None, query=None, **kw):
        """ Select a subset from the current dataset based on multiple criteria. """
        self.logger.debug("selecting a subset of %s based on query '%s'..." % (self.basename, query))
        ds2 = self.__class__(name2)
        ds2._metadata['sources'] = src = self._metadata['sources'][:]
        _tmp = {s: 0 for s in src}
        for e in self._filter(query, **kw):
            for s in src:
                if e.realpath.startswith(s):
                    _tmp[s] += 1
                    break
            ds2[Executable(dataset=self, dataset2=ds2, hash=e.hash)] = self[e.hash, True]
        for s, cnt in _tmp.items():
            if cnt == 0:
                src.remove(s)
        ds2._save()
    
    def show(self, limit=10, per_format=False, **kw):
        """ Show an overview of the dataset. """
        self.__limit = limit if limit > 0 else len(self)
        self.__per_format = per_format
        if len(self) > 0:
            c = List(["**#Executables**: %d" % self._metadata['executables'],
                      "**Format(s)**:    %s" % ", ".join(self._metadata['formats']),
                      "**Packer(s)**:    %s" % ", ".join(self._metadata['counts'].keys()),
                      "**Size**:         %s" % ts.human_readable_size(self.path.size),
                      "**With files**:   %s" % ["no", "yes"][self._files]])
            r = Report(Section("Dataset characteristics"), c)
            r.extend(self.overview)
            print(mdv.main(r.md()))
        else:
            self.logger.warning("Empty dataset")
    
    @backup
    def update(self, source_dir=".", formats=["All"], labels=None, detect=False, **kw):
        """ Update the dataset with a folder of binaries, detecting used packers if 'detect' is set to True, otherwise
             packing randomly. If labels are provided, they are used instead of applying packer detection. """
        l, self.formats = self.logger, formats
        labels = Dataset.labels_from_file(labels)
        if len(labels) == 0 and not detect:
            if ts.confirm("No label was provided ; consider every executable as not packed ?"):
                labels = None
            else:
                l.warning("No label provided, cannot proceed.")
                return
        if source_dir is None:
            l.warning("No source folder provided")
            return
        if not isinstance(source_dir, list):
            source_dir = [source_dir]
        l.info("Source directories: %s" % ",".join(map(str, set(source_dir))))
        self._metadata.setdefault('formats', [])
        self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + source_dir)))
        i, n, pbar = -1, sum(1 for _ in self._walk(True, {'All': source_dir}, True)), None
        for i, e in enumerate(self._walk(True, {'All': source_dir})):
            # set the progress bar now to not overlap with self._walk's logging
            if pbar is None:
                pbar = tqdm(total=n, unit="executable")
            if e.format not in self._metadata['formats']:
                self._metadata['formats'].append(e.format)
            # precedence: input dictionary of labels > dataset's own labels > detection (if enabled) > discard
            try:
                self[e] = None if labels is None else labels[e.hash]
            except KeyError:
                if detect:
                    self[e] = Detector.detect(e)
                else:
                    del self[e]  # ensure there is no trace of this executable to be discarded
            pbar.update()
        if pbar:
            pbar.close()
        if i < 0:
            l.warning("No executable found")
        self._save()
    
    def view(self, query=None, **kw):
        """ View executables from the dataset given multiple criteria. """
        src = self._metadata.get('sources', [])
        def _shorten(path):
            p = ts.Path(path)
            for i, s in enumerate(src):
                if p.is_under(s):
                    return i, str(p.relative_to(s))
            return -1, path
        # prepare the table of records
        d, h = [], ["Hash", "Path", "Size", "Creation", "Modification", "Label"]
        for e in self._filter(query, **kw):
            e = Executable(dataset=self, hash=e.hash)
            i, p = _shorten(e.realpath)
            if i >= 0:
                p = "[%d]/%s" % (i, p)
            d.append([e.hash, p, ts.human_readable_size(e.size), e.ctime.strftime("%d/%m/%y"),
                      e.mtime.strftime("%d/%m/%y"), "" if str(e.label) in ["nan", "None"] else e.label])
        if len(d) == 0:
            return
        r = [Text("Sources:\n%s" % "\n".join("[%d] %s" % (i, s) for i, s in enumerate(src))),
             Table(d, title="Filtered records", column_headers=h)]
        print(mdv.main(Report(*r).md()))
    
    @property
    def backup(self):
        """ Get the latest backup. """
        tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
        for backup in sorted(tmp.listdir(self.__class__.check), key=lambda p: -int(p.basename)):
            return self.__class__(backup)
    
    @backup.setter
    def backup(self, dataset):
        """ Make a backup copy. """
        if len(self._data) == 0:
            return
        tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
        backups, i = [], 0
        for i, backup in enumerate(sorted(tmp.listdir(Dataset.check), key=lambda p: -int(p.basename))):
            backup, n = self.__class__(backup), 0
            # if there is a change since the last backup, create a new one
            if i == 0 and dataset != backup:
                dataset._copy(tmp.joinpath(str(int(time.time()))))
                n = 1
            elif i >= BACKUP_COPIES - n:
                backup._remove()
        if i == 0:
            dataset._copy(tmp.joinpath(str(int(time.time()))))
    
    @property
    def files(self):
        """ Get the Path instance for the 'files' folder. """
        if self._files or self.__class__ is Dataset:
            return self.path.joinpath("files")
        raise AttributeError("'FilelessDataset' object has no attribute 'files'")
    
    @property
    def labels(self):
        """ Get the series of labels. """
        return self._data.label
    
    @property
    def basename(self):
        """ Dummy shortcut for dataset's path.basename. """
        return self.path.basename
    
    @property
    def name(self):
        """ Get the name of the dataset composed with its list of formats. """
        fmt, p = getattr(self, "formats", []), self.path.basename
        return "%s(%s)" % (p, ",".join(sorted(collapse_formats(*fmt)))) if len(fmt) > 0 else self.path.basename
    
    @property
    def overview(self):
        """ Represent an overview of the dataset. """
        self.logger.debug("computing dataset overview...")
        r = []
        CAT = ["<20kB", "20-50kB", "50-100kB", "100-500kB", "500kB-1MB", ">1MB"]
        size_cat = lambda s: CAT[0] if s < 20 * 1024 else CAT[1] if 20 * 1024 <= s < 50 * 1024 else \
                             CAT[2] if 50 * 1024 <= s < 100 * 1024 else CAT[3] if 100 * 1024 <= s < 500 * 1024 else \
                             CAT[4] if 500 * 1024 <= s < 1024 * 1024 else CAT[5]
        data1, data2 = {}, {}
        formats = expand_formats("All") if self.__per_format else ["All"]
        src = self._metadata.get('sources', [])
        r = []
        def _shorten(path):
            p = ts.Path(path)
            for i, s in enumerate(src):
                if p.is_under(s):
                    return i, str(p.relative_to(s))
            return -1, path
        # parse formats, collect counts per size range and list of files
        for fmt in formats:
            d = {c: [0, 0] for c in CAT}
            for e in self:
                if fmt != "All" and e.format != fmt:
                    continue
                l, s = "" if str(e.label) in ["", "nan", "None"] else e.label, size_cat(e.size)
                d[s][0] += 1
                if l != "":
                    d[s][1] += 1
                data2.setdefault(fmt, [])
                if len(data2[fmt]) < self.__limit:
                    i, p = _shorten(e.realpath)
                    if i >= 0:
                        p = "[%d]/%s" % (i, p)
                    row = [e.hash, p, e.ctime.strftime("%d/%m/%y"), e.mtime.strftime("%d/%m/%y"), l]
                    data2[fmt].append(row)
                elif len(data2[fmt]) == self.__limit:
                    data2[fmt].append(["...", "...", "...", "...", "..."])
            total, totalp = sum([v[0] for v in d.values()]), sum([v[1] for v in d.values()])
            if total == 0:
                continue
            data1.setdefault(fmt, [])
            for c in CAT:
                data1[fmt].append([c, d[fmt][0], "%.2f" % (100 * (float(d[c][0]) / total)) if total > 0 else 0,
                                      d[fmt][1], "%.2f" % (100 * (float(d[c][1]) / totalp)) if totalp > 0 else 0])
            data1[fmt].append(["Total", str(total), "", str(totalp), ""])
        # display statistics if any
        if len(data1) > 0:
            r.append(Section("Executables per size and format"))
            for fmt in formats:
                fmt = fmt if self.__per_format else "All"
                if fmt in data1:
                    if fmt != "All":
                        r.append(Subsection(fmt))
                    # if all packed
                    if all(x1 == x2 for x1, x2 in [(r[2], r[4]) for r in data1[fmt]]):
                        _t, h = "packed", ["Size Range", "Packed", "%"]
                    # none packed
                    elif data1[fmt][-1][3] == "0":
                        _t, h = "not-packed", ["Size Range", "Not Packed", "%"]
                    # mix of not packed and packed
                    else:
                        _t, h = "mix", ["Size Range", "Total", "%", "Packed", "%"]
                    d = data1[fmt] if _t == "mix" else \
                        [[r[0], r[3], r[4]] for r in data1[fmt]] if _t == "packed" else \
                        [r[:3] for r in data1[fmt]]
                    r += [Table(d, title=fmt, column_headers=h)]
                if fmt == "All":
                    break
        r.append(Rule())
        r.append(Text("**Sources**:\n\n%s" % "\n".join("[%d] %s" % (i, s) for i, s in enumerate(src))))
        # display files if any
        if len(data2) > 0:
            r.append(Section("Executables per label and format"))
            for fmt in formats:
                fmt = fmt if self.__per_format else "All"
                if fmt in data2:
                    if fmt != "All":
                        r.append(Subsection(fmt))
                    r += [Table(data2[fmt], title=fmt,
                                column_headers=["Hash", "Path", "Creation", "Modification", "Label"])]
                if fmt == "All":
                    break
        return r
    
    @classmethod
    def check(cls, name):
        try:
            cls.validate(name, False)
            return True
        except ValueError as e:
            return False
    
    @classmethod
    def iteritems(cls):
        s = cls.summarize(str(config['datasets']), False)
        if len(s) > 0:
            for row in s[1].data:
                yield Dataset(row[0])
    
    @classmethod
    def validate(cls, name, load=True):
        f = getattr(cls, "_files", True)
        ds = cls(name, load=False)
        p = ds.path
        if not p.is_dir():
            raise ValueError
        if f and not p.joinpath("files").is_dir() or not f and p.joinpath("files").is_dir():
            raise ValueError
        for fn in ["data.csv", "metadata.json"] + [["features.json"], []][f]:
            if not p.joinpath(fn).exists():
                raise ValueError
        if load:
            ds._load()
        return ds
    
    @staticmethod
    def labels_from_file(labels):
        labels = labels or {}
        if isinstance(labels, str):
            labels = ts.Path(labels)
        if isinstance(labels, ts.Path) and labels.is_file():
            with labels.open() as f:
                labels = json.load(f)
        if not isinstance(labels, dict):
            raise ValueError("Bad labels ; not a dictionary or JSON file")
        return labels
    
    @staticmethod
    def summarize(path=None, show=False, hide_files=False):
        datasets = []
        headers = ["Name", "#Executables", "Size"] + [["Files"], []][hide_files] + ["Formats", "Packers"]
        for dset in ts.Path(config['datasets']).listdir(lambda x: x.joinpath("metadata.json").exists()):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                row = [
                    dset.basename,
                    str(metadata['executables']),
                    ts.human_readable_size(dset.size),
                ] + [[["no", "yes"][dset.joinpath("files").exists()]], []][hide_files] + [
                    ",".join(sorted(metadata['formats'])),
                    shorten_str(",".join("%s{%d}" % i \
                                for i in sorted(metadata['counts'].items(), key=lambda x: (-x[1], x[0])))),
                ]
            except Exception as err:
                row = None
                if show:
                    if headers[-1] != "Reason":
                        headers.append("Reason")
                    row = [
                        dset.basename,
                        str(metadata.get('executables', colored("?", "red"))),
                        ts.human_readable_size(dset.size),
                    ] + [[["no", "yes"][dset.joinpath("files").exists()]], []][hide_files] + [
                        colored("?", "red"), colored("?", "red"),
                        colored("%s: %s" % (err.__class__.__name__, str(err)), "red")
                    ]
            if row:
                datasets.append(row)
        n = len(datasets)
        return [] if n == 0 else [Section("Datasets (%d)" % n), Table(datasets, column_headers=headers)]

