# -*- coding: UTF-8 -*-
import pandas as pd
import re
from datetime import datetime, timedelta
from textwrap import wrap
from tinyscript import b, colored, hashlib, json, logging, random, subprocess, time, ts
from tinyscript.report import *
from tqdm import tqdm

from .config import config
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
        self._files = getattr(self.__class__, "_files", True)
        self.path = ts.Path(config['datasets'].joinpath(name), create=load).absolute()
        self.sources = source_dir or PACKING_BOX_SOURCES
        if isinstance(self.sources, list):
            self.sources = {'All': self.sources}
        for _, sources in self.sources.items():
            for source in sources[:]:
                s = ts.Path(source, expand=True)
                if not s.exists() or not s.is_dir():
                    sources.remove(source)
        if load:
            self._load()
        self.categories = getattr(self, "_metadata", {}).get('categories', [])
        self.__change = False
        self.__limit = 20
        self.__per_category = False
    
    def __delitem__(self, executable):
        """ Remove an executable (by its real name or hash) from the dataset. """
        self.__change = True
        h = executable
        # first, ensure we handle the hash name (not the real one)
        try:
            h = self._data.loc[df['realpath'] == h, 'hash'].iloc[0]
        except:
            pass
        if len(self._data) > 0:
            self.logger.debug("removing %s..." % h)
            self._data = self._data[self._data.hash != h]
        if self._files:
            self.files.joinpath(h).remove(error=False)
    
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
    
    def __setattr__(self, name, value):
        if name == "categories":
            # get the list of packers related to the selected categories
            self._categories_exp = expand_categories(*value)
            self.packers = [p for p in Packer.registry if p.status > 2 and p.check(*self._categories_exp)]
            if len(self.packers) == 0:
                raise ValueError("No packer found for these categories")
        super(Dataset, self).__setattr__(name, value)
    
    def __setitem__(self, executable, label):
        """ Add an executable based on its real name to the dataset.
        
        :param executable: either the path to the executable or its Executable instance
        :param label:      either the text label of the given executable or its dictionary of data
        """
        self.__change = True
        e = executable
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
        self._metadata.setdefault('categories', [])
        if e.category not in self._metadata['categories']:
            self._metadata['categories'].append(e.category)
        if len(self._data) > 0 and e.hash in list(self._data.hash):
            self.logger.debug("updating %s..." % e.hash)
            for n, v in d.items():
                self._data.at[self._data.hash == e.hash, n] = v
        else:
            self.logger.debug("adding %s..." % e.hash)
            self._data = self._data.append(d, ignore_index=True)
    
    def _copy(self, path):
        """ Copy the current dataset to a given destination. """
        self.path.copy(path)
    
    def _filter(self, query=None, **kw):
        """ Yield executables' hashes from the dataset using Pandas' query language. """
        i = -1
        for i, row in enumerate(self._data.query(query or "tuple()").itertuples()):
            yield row
        if i == -1:
            self.logger.warning("No data selected")
    
    def _load(self):
        """ Load dataset's associated files or create them. """
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
        self.path.remove(error=False)
    
    def _save(self):
        """ Save dataset's state to JSON files. """
        if not self.__change:
            return
        self._metadata['categories'] = sorted(collapse_categories(*self._metadata['categories']))
        self._metadata['counts'] = self._data.label.value_counts().to_dict()
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
        m = 0
        candidates = []
        packers = [p.name for p in Packer.registry]
        for cat, srcs in (sources or self.sources).items():
            if all(c not in expand_categories(cat) for c in self._categories_exp):
                continue
            for src in srcs:
                for exe in ts.Path(src, expand=True).walk():
                    if not exe.is_file():
                        continue
                    exe = Executable(exe, dataset=self)
                    if exe.category not in self._categories_exp or exe.filename in packers:
                        continue  # ignore unrelated files and packers themselves
                    if walk_all:
                        yield exe
                    else:
                        candidates.append(exe)
        if len(candidates) > 0:
            random.shuffle(candidates)
            for c in candidates:
                yield c
    
    def edit(self, **kw):
        """ Edit the data CSV file. """
        subprocess.call(["vd", str(self.path.joinpath("data.csv").absolute()), "--csv-delimiter", "\";\""],
                        stderr=subprocess.PIPE)
    
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
    
    def list(self, show_all=False, **kw):
        """ List all the datasets from the given path. """
        self.logger.debug("summarizing datasets from %s..." % config['datasets'])
        d = Dataset.summarize(str(config['datasets']), show_all)
        if len(d) > 0:
            print(mdv.main(Report(*d).md()))
        else:
            self.logger.warning("No dataset found in workspace (%s)" % config['datasets'])
    
    @backup
    def make(self, n=100, categories=["All"], balance=False, packer=None, refresh=False, **kw):
        """ Make n new samples in the current dataset among the given binary categories, balanced or not according to
             the number of distinct packers. """
        pbar = tqdm(total=n, unit="executable")
        self.categories = categories
        sources = []
        for cat, src in self.sources.items():
            if all(c not in expand_categories(cat) for c in self._categories_exp):
                continue
            sources.extend(src)
        self.logger.info("Source directories:    %s" % ",".join(set(sources)))
        self.logger.info("Considered categories: %s" % ",".join(categories))
        self.logger.info("Selected packers:      %s" % ",".join(["All"] if packer is None else \
                                                                [p.__class__.__name__ for p in packer]))
        self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + sources)))
        # get executables to be randomly packed or not
        i = 0
        for exe in self._walk(n <= 0):
            if i >= n > 0:
                break
            self.logger.debug("handling %s..." % exe)
            packers = [p for p in (packer or Packer.registry) if p in self.packers]
            if exe.destination.exists():
                if refresh:
                    self[exe, True] = None  # this won't overwrite the label
                continue
            i += 1
            label = short_label = None
            if random.randint(0, len(packers) if balance else 1):
                if len(packers) == 0:
                    self.logger.critical("No packer left")
                    return
                random.shuffle(packers)
                for p in packers:
                    label = p.pack(str(exe.absolute()))
                    if not label or p._bad:
                        if label is False or p._bad:
                            self.logger.warning("Disabling %s..." % p.__class__.__name__)
                            self.packers.remove(p)
                            label = None
                        continue
                    else:  # consider short label (e.g. "midgetpack", not "midgetpack[<password>]")
                        short_label = label.split("[")[0]
                    break
            self[exe] = short_label
            pbar.update()
        if i < n - 1:
            self.logger.warning("Found too few candidate executables")
        l = len(self)
        if l > 0:
            # finally, save dataset's metadata and labels to JSON files
            p = sorted(list(set([l for l in self._data.label.values if isinstance(l, str)])))
            self.logger.info("Used packers: %s" % ", ".join(p))
            self.logger.info("#Executables: %d" % l)
        self._save()
    
    def purge(self, **kw):
        """ Truncate and recreate a blank dataset. """
        self.logger.debug("purging %s..." % self.path)
        self._remove()
        # also recursively purge the backups
        try:
            self.backup.purge()
        except AttributeError:
            pass
    
    @backup
    def remove(self, query=None, **kw):
        """ Remove executables from the dataset given multiple criteria. """
        for e in self._filter(query, **kw):
            del self[e.hash]
        self._save()
    
    def rename(self, name2=None, **kw):
        """ Rename the current dataset. """
        path2 = config['datasets'].joinpath(name2)
        if not path2.exists():
            self.logger.debug("renaming dataset (and backups) to %s..." % path2)
            tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
            self.path = self.path.rename(path2)
            tmp.rename(ts.TempPath(".dataset-backup", hex(hash(self))[2:]))
        else:
            self.logger.warning("%s already exists" % path2)
    
    def revert(self, **kw):
        """ Revert to the latest version of the dataset (if a backup copy exists in /tmp). """
        b = self.backup
        if b is None:
            self.logger.warning("No backup found ; could not revert")
        else:
            self.logger.debug("reverting dataset to previous backup...")
            self._remove()
            b._copy(self.path)
            b._remove()
            self._save()
    
    def select(self, name2=None, query=None, **kw):
        """ Select a subset from the current dataset based on multiple criteria. """
        ds2 = Dataset(name2)
        for e in self._filter(query, **kw):
            ds2[Executable(dataset=self, hash=e.hash)] = self[e.hash, True]
        ds2._save()
    
    def show(self, limit=10, per_category=False, **kw):
        """ Show an overview of the dataset. """
        self.__limit = limit if limit > 0 else len(self)
        self.__per_category = per_category
        if len(self) > 0:
            c = List(["**#Executables**: %d" % self._metadata['executables'],
                      "**Categories**:   %s" % ", ".join(self._metadata['categories']),
                      "**Packers**:      %s" % ", ".join(self._metadata['counts'].keys()),
                      "**Size**:         %s" % ts.human_readable_size(self.path.size),
                      "**With files**:   %s" % ["no", "yes"][self._files]])
            r = Report(Section("Dataset characteristics"), c)
            r.extend(self.overview)
            print(mdv.main(r.md()))
        else:
            self.logger.warning("Empty dataset")
    
    @backup
    def update(self, source_dir=".", categories=["All"], labels=None, detect=False, **kw):
        """ Update the dataset with a folder of binaries, detecting used packers if 'detect' is set to True, otherwise
             packing randomly. If labels are provided, they are used instead of applying packer detection. """
        self.categories = categories
        labels = Dataset.labels_from_file(labels)
        if source_dir is None:
            self.logger.warning("No source folder provided")
            return self
        if not isinstance(source_dir, list):
            source_dir = [source_dir]
        self._metadata.setdefault('categories', [])
        self._metadata['sources'] = list(set(map(str, self._metadata.get('sources', []) + source_dir)))
        n = 0
        for e in self._walk(True, {'All': source_dir}, True):
            n += 1
        pbar = tqdm(total=n, unit="executable")
        i = -1
        for i, e in enumerate(self._walk(True, {'All': source_dir})):
            if e.category not in self._metadata['categories']:
                self._metadata['categories'].append(e.category)
            # precedence: input dictionary of labels > dataset's own labels > detection (if enabled) > discard
            try:
                self[e] = labels[e.hash]
            except KeyError:
                if detect:
                    self[e] = Detector.detect(e)
                else:
                    del self[e]  # ensure there is no trace of this executable to be discarded
            pbar.update()
        if i < 0:
            self.logger.warning("No executable found")
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
        r = [Text("Sources:\n %s" % "\n ".join("[%d] %s" % (i, s) for i, s in enumerate(src))),
             Table(d, title="Filtered records", column_headers=h)]
        print(mdv.main(Report(*r).md()))
    
    @property
    def backup(self):
        """ Get the latest backup. """
        tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
        for backup in sorted(tmp.listdir(Dataset.check), key=lambda p: -int(p.basename)):
            return Dataset(backup)
    
    @backup.setter
    def backup(self, dataset):
        """ Make a backup copy. """
        if len(self._data) == 0:
            return
        tmp = ts.TempPath(".dataset-backup", hex(hash(self))[2:])
        backups, i = [], 0
        for i, backup in enumerate(sorted(tmp.listdir(Dataset.check), key=lambda p: -int(p.basename))):
            backup = Dataset(backup)
            if i == 0 and dataset != backup:
                dataset._copy(tmp.joinpath(str(int(time.time()))))
            elif i >= BACKUP_COPIES:
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
    def name(self):
        """ Get the name of the dataset composed with its list of categories. """
        return "%s(%s)" % (self.path.basename, ",".join(sorted(collapse_categories(*self.categories)))) \
               if len(self.categories) > 0 else self.path.basename
    
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
        categories = expand_categories("All") if self.__per_category else ["All"]
        for category in categories:
            d = {c: [0, 0] for c in CAT}
            for e in self:
                if category != "All" and e.category != category:
                    continue
                l, s = "" if str(e.label) in ["", "nan", "None"] else e.label, size_cat(e.size)
                d[s][0] += 1
                if l != "":
                    d[s][1] += 1
                data2.setdefault(category, [])
                if len(data2[category]) < self.__limit:
                    row = [e.hash, e.realpath, e.ctime.strftime("%d/%m/%y"), e.mtime.strftime("%d/%m/%y"), l]
                    data2[category].append(row)
                elif len(data2[category]) == self.__limit:
                    data2[category].append(["...", "...", "...", "...", "..."])
            total, totalp = sum([v[0] for v in d.values()]), sum([v[1] for v in d.values()])
            if total == 0:
                continue
            data1.setdefault(category, [])
            for c in CAT:
                data1[category].append([c, d[c][0], "%.2f" % (100 * (float(d[c][0]) / total)) if total > 0 else 0,
                                        d[c][1], "%.2f" % (100 * (float(d[c][1]) / totalp)) if totalp > 0 else 0])
            data1[category].append(["Total", str(total), "", str(totalp), ""])
        if len(data1) > 0:
            headers = ["Size Range", "Total", "%", "Packed", "%"]
            r.append(Section("Executables per size and category"))
            for c in categories:
                c = c if self.__per_category else "All"
                if c in data1:
                    if c != "All":
                        r.append(Subsection(c))
                    r += [Table(data1[c], title=c, column_headers=headers)]
                if c == "All":
                    break
        if len(data2) > 0:
            r.append(Section("Executables per label and category"))
            for c in categories:
                c = c if self.__per_category else "All"
                if c in data2:
                    if c != "All":
                        r.append(Subsection(c))
                    r += [Table(data2[c], title=c,
                                column_headers=["Hash", "Path", "Creation", "Modification", "Label"])]
                if c == "All":
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
    def summarize(path=None, show=False):
        datasets = []
        headers = ["Name", "#Executables", "Size", "Files", "Categories", "Packers"]
        for dset in ts.Path(config['datasets']).listdir(lambda x: x.joinpath("metadata.json").exists()):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                datasets.append([
                    dset.basename,
                    str(metadata['executables']),
                    ts.human_readable_size(dset.size),
                    ["no", "yes"][dset.joinpath("files").exists()],
                    ",".join(sorted(metadata['categories'])),
                    ",".join("%s{%d}" % i for i in sorted(metadata['counts'].items(), key=lambda x: -x[1])),
                ])
            except KeyError:
                if show:
                    datasets.append([
                        dset.basename,
                        str(metadata.get('executables', colored("?", "red"))),
                        colored("corrupted", "red"),
                    ])
        n = len(datasets)
        return [] if n == 0 else [Section("Datasets (%d)" % n), Table(datasets, column_headers=headers)]

