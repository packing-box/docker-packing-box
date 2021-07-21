# -*- coding: UTF-8 -*-
import pandas as pd
import re
from datetime import datetime, timedelta
from tinyscript import b, colored, hashlib, json, logging, random, time, ts
from tinyscript.report import *
from tqdm import tqdm

from .executable import Executable
from .detector import Detector
from .packer import Packer
from ..utils import *


__all__ = ["Dataset", "PACKING_BOX_SOURCES"]


BACKUP_COPIES = 3
PACKING_BOX_SOURCES = {
    'ELF': ["/sbin", "/usr/bin"],
    'PE':  ["/root/.wine/drive_c/windows", "/root/.wine32/drive_c/windows"],
}


def backup(f):
    """ Simple method decorator for making a backup of the dataset. """
    def _wrapper(s, *a, **kw):
        s.backup = s
        return f(s, *a, **kw)
    return _wrapper


class Dataset:
    """ Folder structure:
    
    [name]
      +-- files
      |     +-- {executables, renamed to their SHA256 hashes}
      +-- data.csv (contains the labels)        # features for an executable, formatted for ML
      +-- features.json                         # dictionary of feature name/description pairs
      +-- labels.json                           # dictionary of hashes with their full labels
      +-- metadata.json                         # simple statistics about the dataset
      +-- names.json                            # dictionary of hashes and their real filenames
    """
    FIELDS = ["category", "ctime", "filetype", "hash", "label", "mtime", "size"]
    
    @logging.bindLogger
    def __init__(self, name="dataset", source_dir=None, load=True, **kw):
        p = config['workspace'].joinpath("datasets")
        p.mkdir(exist_ok=True)
        self.path = ts.Path(p.joinpath(name), create=True).absolute()
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
    
    def __delitem__(self, executable):
        """ Remove an executable (by its real name or hash) from the dataset. """
        self.__change = True
        e = executable
        try:
            e, refresh = e
        except (TypeError, ValueError):
            refresh = False
        # first, ensure we handle the hash name (not the real one)
        h = {n: h for h, n in self._names.items()}.get(e, e)
        if not refresh and h in self._labels:
            self.logger.debug("removing %s..." % h)
        if len(self) > 0:
            self._data = self._data[self._data.hash != h]
        # then try removing
        for l in ["_labels", "_names"]:
            try:
                del getattr(self, l)[h]
            except KeyError:
                pass
        self.files.joinpath(h).remove(error=False)
    
    def __eq__(self, dataset):
        """ Custom equality function. """
        ds = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        # NB: we don't care of inconsistency in the features dictionaries
        return all(getattr(self, attr) == getattr(ds, attr) for attr in ["_metadata", "_labels", "_names"]) \
               and self._data.equals(ds._data) and list(self.files.listdir()) == list(ds.files.listdir())
    
    def __getitem__(self, hash):
        """ Get a data row related to the given hash from the dataset. """
        try:
            h, with_headers = hash
        except ValueError:
            h, with_headers = hash, False
        if len(self) == 0:
            raise KeyError(h)
        try:
            row = self._data[self._data.hash == h].iloc[0]
            return row.to_dict() if with_headers else row.to_list()
        except IndexError:
            raise KeyError(h)
    
    def __iter__(self):
        """ Iterate over the dataset. """
        for h, n in self._names.items():
            yield self[h, True], n
    
    def __len__(self):
        """ Get dataset's length. """
        ld, ll, ln = len(self._data), len(self._labels), len(self._names)
        if not ld == ll == ln:
            raise InconsistentDataset("%s (data: %d ; labels: %d ; names: %d)" % (self.name, ld, ll, ln))
        return len(self._labels)
    
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
        try:
            e, refresh = e
        except (TypeError, ValueError):
            refresh = False
        e = Executable(e, dataset=self)
        if refresh or not e.is_file():
            del self[e, refresh]
            return
        if isinstance(label, dict):
            d = label
            for k in []:
                if d.get(k):
                    setattr(e, k, d[k])
            e.data = {k: v for k, v in d.items() if k not in Dataset.FIELDS}
            label = e.label
            for k in Dataset.FIELDS:
                if k not in d:
                    d[k] = getattr(e, k)
        else:
            e.label = label
            d = {k: getattr(e, k) for k in Dataset.FIELDS}
            d.update(e.data)
            e.copy()
        self._features.update(e.features)
        self._metadata.setdefault('categories', [])
        if e.category not in self._metadata['categories']:
            self._metadata['categories'].append(e.category)
        self._names[e.hash] = str(e.realpath)
        if len(self._data) > 0 and e.hash in list(self._data.hash):
            self.logger.debug("updating %s..." % e.hash)
            self._data.update(pd.DataFrame(d, index=[0]))
        else:
            self.logger.debug("adding %s..." % e.hash)
            self._data = self._data.append(d, ignore_index=True)
        if not refresh or e.hash not in self._labels:
            self._labels[e.hash] = label
    
    def _copy(self, path):
        """ Copy the current dataset to a given destination. """
        self.path.copy(path)
    
    def _filter(self, query=None, **kw):
        """ Yield executables' hashes from the dataset using Pandas' query language. """
        # datetime transformation function for the other formats
        def _time(v, end=False):
            if v is None:
                return [datetime.min, datetime.max][end]
            # year format ; 2020 => start: 01/01/20 ; end: 31/12/20
            try:
                dt = datetime.strptime(v, "%Y")
                return dt.replace(year=dt.year+1) - timedelta(seconds=1) if end else dt
            except ValueError:
                pass
            # year-month formats ; Jan 2020 => start: 01/01/20 ; end: 31/01/20
            for f in ["%Y-%m", "%b %Y", "%B %Y"]:
                try:
                    dt = datetime.strptime(v, f)
                    return dt.replace(month=dt.month+1) - timedelta(seconds=1) if end else dt
                except ValueError:
                    pass
            # year-month-day formats ; 01/01/20 => start: 01/01/20 00:00:00 ; end: 01/01/20 23:59:59
            for f in ["%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d", "%b %d %Y", "%B %d, %Y"]:
                try:
                    dt = datetime.strptime(v, f)
                    return dt.replace(day=dt.day+1) - timedelta(seconds=1) if end else dt
                except:
                    pass
            raise ValueError("Input datetime could not be parseds")
        # set the list of categories
        categories = expand_categories(*(category or ["All"]))
        # format ctime and mtime datetimes for the time criteria
        if ctime_year is not None:
            if not re.match("[1-9][0-9]{3}$", ctime_year):
                raise ValueError("Bad input year")
            ctime_s, ctime_e = _time(ctime_year), _time(ctime_year, True)
        else:
            ctime_s = datetime.min if ctime_start is None else _time(ctime_start)
            ctime_e = datetime.max if ctime_end is None else _time(ctime_end, True)
        if mtime_year is not None:
            if not re.match("[1-9][0-9]{3}$", mtime_year):
                raise ValueError("Bad input year")
            mtime_s, mtime_e = _time(mtime_year), _time(mtime_year, True)
        else:
            mtime_s = datetime.min if mtime_start is None else _time(mtime_start)
            mtime_e = datetime.max if mtime_end is None else _time(mtime_end, True)
        # prepare the list of valid hashes
        hashes = sorted(set(self._labels.keys()) if hash is None else [hash] if not isinstance(hash, list) else hash)
        hashes = hashes[hashes.index(hash_start or hashes[0]):hashes.index(hash_end or hashes[-1])+1]
        # prepare the list of valid labels
        labels = set(self._labels.values()) if label is None else [label] if not isinstance(label, list) else label
        labels = [None if l.lower() == "none" else l for l in labels]
        # now yield executables given filters' intersection
        for h in self._labels.keys():
            e = Executable(dataset=self, hash=h)
            if h in hashes and e.label in labels and ctime_s <= e.ctime <= ctime_e and mtime_s <= e.mtime <= mtime_e \
               and e.category in categories:
                yield e
    
    def _hash(self):
        """ Custom object hashing function. """
        return hashlib.md5(b(str(self.path.resolve()))).hexdigest()
    
    def _load(self):
        """ Load dataset's associated files or create them. """
        self.files.mkdir(exist_ok=True)
        for n in ["data", "features", "labels", "metadata", "names"]:
            p = self.path.joinpath(n + [".json", ".csv"][n == "data"])
            if n == "data":
                try:
                    self._data = pd.read_csv(str(p), sep=";")
                except (OSError, KeyError):
                    self._data = pd.DataFrame()
            elif p.exists():
                with p.open() as f:
                    setattr(self, "_" + n, json.load(f))
            else:
                setattr(self, "_" + n, {})
                p.write_text("{}")
        return self
    
    def _remove(self):
        """ Remove the current dataset. """
        self.path.remove(error=False)
    
    def _save(self):
        """ Save dataset's state to JSON files. """
        if not self.__change:
            return self
        if len(self) > 0:
            self._metadata['counts'] = self._data.label.value_counts().to_dict()
            self._metadata['executables'] = len(self._labels)
            for n in ["data", "features", "labels", "metadata", "names"]:
                if n == "data":
                    self._data = self._data.sort_values("hash")
                    h = ["hash"] + \
                        sorted([c for c in self._data.columns if c in set(Dataset.FIELDS) - set(["hash", "label"])]) + \
                        sorted([c for c in self._data.columns if c not in Dataset.FIELDS]) + ["label"]
                    self._data.to_csv(str(self.path.joinpath("data.csv")), sep=";", columns=h, index=False, header=True)
                else:
                    with self.path.joinpath(n + ".json").open('w+') as f:
                        json.dump(getattr(self, "_" + n), f, indent=2)
        else:
            self.logger.warning("No data to save")
        self.__change = False
    
    def _walk(self, walk_all=False, sources=None):
        """ Walk the sources for random in-scope executables. """
        self.logger.info("Searching for executables...")
        m = 0
        candidates = []
        packers = [p.name for p in Packer.registry]
        for cat, srcs in (sources or self.sources).items():
            if all(c not in expand_categories(cat) for c in self._categories_exp):
                continue
            for src in srcs:
                for exe in ts.Path(src, expand=True).walk():
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
    
    @backup
    def fix(self, labels=None, detect=False, files=False, **kw):
        """ Make dataset's structure and files match. """
        labels = Dataset.labels_from_file(labels)
        # first, ensure there is no duplicate
        self._data = self._data.drop_duplicates()
        if files:
            # fix wrt existing files
            for f in self.files.listdir():
                h = f.filename
                if h not in self._names:
                    del self[h]
                else:
                    e = Executable(self._names[h])
                    self[self._names[h], True] = Detector.detect(e) if detect else \
                                                 labels.get(h, labels.get(self._names[h], self._labels.get(h)))
        else:
            # fix wrt existing labels
            for h, l in list(self._labels.items()):
                e = self.files.joinpath(h)
                if not e.exists() or h not in self._names:
                    del self[h]
                else:
                    self[Executable(dataset=self, hash=h), True] = Detector.detect(e) if detect else l
        self._save()
    
    def is_valid(self):
        """ Check if this Dataset instance has a valid structure. """
        return Dataset.check(self.path)
    
    def list(self, show_all=False, **kw):
        """ List all the datasets from the given path. """
        print(mdv.main(Report(*Dataset.summarize(str(config['workspace'].joinpath("datasets")), show_all)).md()))
    
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
        self._metadata['sources'] = list(set(self._metadata.get('sources', []) + sources))
        # get executables to be randomly packed or not
        i = 0
        for exe in self._walk(n <= 0):
            if i >= n > 0:
                break
            self.logger.debug("handling %s..." % exe)
            packers = [p for p in (packer or Packer.registry) if p in self.packers]
            if exe.destination.exists():
                # when refresh=True, the features are recomputed for the existing target executable ; it allows to
                #  recompute features for a previously made dataset if the list of features was updated
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
    
    @backup
    def merge(self, path2=None, **kw):
        """ Merge another dataset with the current one. """
        ds2 = Dataset(path2)
        # add rows from the input dataset
        for row, name in ds2:
            self[name, False] = row
        # as the previous operation does not update categories and features, do it manually
        self._metadata.setdefault('categories', [])
        for category in ds2._metadata.get('categories', []):
            if category not in self._metadata['categories']:
                self._metadata['categories'].append(category)
        d = {k: v for k, v in ds2._features.items()}
        d.update(self._features)
        self._features = d
        # now copy files from the input dataset
        for f in ds2.files.listdir():
            e = Executable(f, dataset=self)
            self[e, True] = e.label
        self._save()
    
    @backup
    def remove(self, query=None, **kw):
        """ Remove executables from the dataset given multiple criteria. """
        r = self._data.query(query or "tuple()")
        if len(r) == 0:
            self.logger.warning("No data selected")
            return
        for i, e in r.iterrows():
            del self[e.hash]
        self._save()
    
    def rename(self, path2=None, **kw):
        """ Rename the current dataset. """
        if not ts.Path(path2).exists():
            tmp = ts.TempPath(".dataset-backup", self._hash())
            self.path = self.path.rename(path2)
            tmp.rename(ts.TempPath(".dataset-backup", self._hash()))
        else:
            self.logger.warning("%s already exists" % path2)
    
    def reset(self, save=True, **kw):
        """ Truncate and recreate a blank dataset. """
        if save:
            self.backup = self
        self._remove()
        ts.Path(self.path, create=True)
        self._load()
        if save:
            self._save()
    
    def revert(self, **kw):
        """ Revert to the latest version of the dataset (if a backup copy exists in /tmp). """
        b = self.backup
        if b is None:
            self.logger.warning("No backup found ; could not revert")
        else:
            self._remove()
            b._copy(self.path)
            b._remove()
            self._save()
    
    def select(self, path2=None, query=None, **kw):
        """ Select a subset from the current dataset based on multiple criteria. """
        if not ts.Path(path2).exists():
            ds2 = Dataset(path2)
            r = self._data.query(query or "tuple()")
            if len(r) == 0:
                self.logger.warning("No data selected")
                return
            for i, e in r.iterrows():
                ds2[Executable(dataset=self, hash=e.hash)] = self[e.hash, True]
            ds2._save()
        else:
            self.logger.warning("%s already exists" % path2)
    
    def show(self, limit=10, per_category=False, **kw):
        """ Show an overview of the dataset. """
        self.__limit = limit if limit > 0 else len(self._labels)
        self.__per_category = per_category
        if len(self) > 0:
            c = List(["**Number of executables**: %d" % self._metadata['executables'],
                      "**Categories**:            %s" % ", ".join(self._metadata['categories']),
                      "**Packers**:               %s" % ", ".join(self._metadata['counts'].keys())])
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
        for e in self._walk(sources={'All': source_dir}):
            if e.category not in self._categories_exp:
                continue
            if e.category not in self._metadata['categories']:
                self._metadata['categories'].append(e.category)
            # precedence: input dictionary of labels > dataset's own labels > detection (if enabled) > no label
            self[e] = labels.get(e.hash, labels.get(str(e), Detector.detect(e) if detect else None))
        self._save()
    
    def view(self, query=None, **kw):
        """ View executables from the dataset given multiple criteria. """
        d = []
        r = self._data.query(query or "tuple()")
        if len(r) == 0:
            self.logger.warning("No data selected")
            return
        for i, e in r.iterrows():
            e = Executable(dataset=self, hash=e.hash)
            d.append([e.hash, str(e), e.ctime.strftime("%d/%m/%y"), e.mtime.strftime("%d/%m/%y"), e.label or ""])
        r = [Table(d, title="Filered records", column_headers=["Hash", "Path", "Creation", "Modification", "Label"])]
        print(mdv.main(Report(*r).md()))
    
    @property
    def backup(self):
        """ Get the latest backup. """
        tmp = ts.TempPath(".dataset-backup", self._hash())
        for backup in sorted(tmp.listdir(Dataset.check), key=lambda p: -int(p.stem)):
            return Dataset(backup)
    
    @backup.setter
    def backup(self, dataset):
        """ Make a backup copy. """
        if len(self) == 0:
            return
        tmp = ts.TempPath(".dataset-backup", self._hash())
        backups, i = [], 0
        for i, backup in enumerate(sorted(tmp.listdir(Dataset.check), key=lambda p: -int(p.stem))):
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
        return self.path.joinpath("files")
    
    @property
    def labels(self):
        """ Get the series of labels. """
        return self._data.label
    
    @property
    def name(self):
        """ Get the name of the dataset composed with its list of categories. """
        return "%s(%s)" % (self.path.stem, ",".join(self.categories))
    
    @property
    def overview(self):
        """ Represent an overview of the dataset. """
        r = []
        CAT = ["<20kB", "20-50kB", "50-100kB", "100-500kB", "500kB-1MB", ">1MB"]
        size_cat = lambda s: CAT[0] if s < 20 * 1024 else CAT[1] if 20 * 1024 <= s < 50 * 1024 else \
                             CAT[2] if 50 * 1024 <= s < 100 * 1024 else CAT[3] if 100 * 1024 <= s < 500 * 1024 else \
                             CAT[4] if 500 * 1024 <= s < 1024 * 1024 else CAT[5]
        data1, data2 = {}, {}
        categories = expand_categories("All") if self.__per_category else ["All"]
        for category in categories:
            d = {c: [0, 0] for c in CAT}
            for h, label in self._labels.items():
                exe = Executable(dataset=self, hash=h)
                if category != "All" and exe.category != category:
                    continue
                s = size_cat(exe.size)
                d[s][0] += 1
                if label is not None:
                    d[s][1] += 1
                data2.setdefault(category, [])
                if len(data2[category]) < self.__limit:
                    row = [v if isinstance(v, str) else "" for k, v in self[h, True].items() if k in ["hash", "label"]]
                    row.insert(1, exe.mtime.strftime("%d/%m/%y"))
                    row.insert(1, exe.ctime.strftime("%d/%m/%y"))
                    row.insert(1, self._names[h])
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
    
    @staticmethod
    def check(folder):
        try:
            Dataset.validate(folder, False)
            return True
        except ValueError as e:
            return False
    
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
        headers = ["Name", "Size", "Categories", "Packers"]
        for dset in ts.Path(config['workspace'].joinpath("datasets")).listdir(Dataset.check):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                datasets.append([
                    dset.stem,
                    str(metadata['executables']),
                    ",".join(sorted(metadata['categories'])),
                    ",".join("%s{%d}" % i for i in sorted(metadata['counts'].items(), key=lambda x: -x[1])),
                ])
            except KeyError:
                if show:
                    datasets.append([
                        dset.stem,
                        str(metadata.get('executables', colored("?", "red"))),
                        colored("corrupted", "red"),
                    ])
        n = len(datasets)
        return [] if n == 0 else [Section("Datasets (%d)" % n), Table(datasets, column_headers=headers)]
    
    @staticmethod
    def validate(name, load=True):
        ds = Dataset(name, load=False)
        p = ds.path
        if not p.exists():
            raise ValueError("Folder does not exist")
        if not p.is_dir():
            raise ValueError("Input is not a folder")
        if not p.joinpath("files").exists():
            raise ValueError("Files subfolder does not exist")
        if not p.joinpath("files").is_dir():
            raise ValueError("Files subfolder is not a folder")
        for fn in ["data.csv", "features.json", "labels.json", "metadata.json", "names.json"]:
            if not p.joinpath(fn).exists():
                raise ValueError("Folder does not have %s" % fn)
        if load:
            ds._load()
        return ds


class InconsistentDataset(ValueError):
    pass

