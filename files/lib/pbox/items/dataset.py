# -*- coding: UTF-8 -*-
import pandas as pd
from tinyscript import b, colored, hashlib, json, logging, random, time, ts
from tinyscript.report import *
from tqdm import tqdm
try:  # from Python3.9
    import mdv3 as mdv
except ImportError:
    import mdv

from .executable import Executable
from .detector import Detector
from .packer import Packer
from ..utils import expand_categories


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
      +-- data.csv (contains the labels)        # features for an executable, formatted for ML
      +-- features.json                         # dictionary of feature name/description pairs
      +-- labels.json                           # dictionary of hashes with their full labels
      +-- metadata.json                         # simple statistics about the dataset
      +-- names.json                            # dictionary of hashes and their real filenames
    """
    @logging.bindLogger
    def __init__(self, path="dataset", source_dir=None, **kw):
        self.path = ts.Path(path, create=True).absolute()
        self.sources = source_dir or PACKING_BOX_SOURCES
        if isinstance(self.sources, list):
            self.sources = {'All': self.sources}
        for _, sources in self.sources.items():
            for source in sources[:]:
                s = ts.Path(source, expand=True)
                if not s.exists() or not s.is_dir():
                    sources.remove(source)
        self._load()
        self.categories = self._metadata.get('categories', [])
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
        if not len(self._data) == len(self._labels) == len(self._names):
            raise InconsistentDataset(self.name)
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
        """ Add an executable based on its real name to the dataset. """
        self.__change = True
        e = executable
        try:
            e, refresh = e
        except (TypeError, ValueError):
            refresh = False
        if not isinstance(e, Executable):
            e = Executable(e)
        if refresh or not e.is_file():
            del self[e, refresh]
            return
        if isinstance(label, dict):
            d = label
            e = Executable(e, hash=d['hash'], data={k: v for k, v in d.items() if k not in ["hash", "label"]})
            label = d['label']
        else:
            d = {'hash': e.hash, 'label': label}
            d.update(e.data)
            self._features.update(e.features)
            e.copy()
            self._metadata.setdefault('categories', [])
            if e.category not in self._metadata['categories']:
                self._metadata['categories'].append(e.category)
        self._names[e.hash] = str(e)
        if len(self._data) > 0 and e.hash in self._data.hash:
            self.logger.debug("updating %s..." % e.hash)
            self._data.update(pd.DataFrame.from_dict(d))
        else:
            self.logger.debug("adding %s..." % e.hash)
            self._data = self._data.append(d, ignore_index=True)
        if not refresh or e.hash not in self._labels:
            self._labels[e.hash] = label
    
    def _copy(self, path):
        """ Copy the current dataset to a given destination. """
        self.path.copy(path)
    
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
            self._metadata['counts'] = self._data['label'].value_counts().to_dict()
            self._metadata['executables'] = len(self._labels)
            for n in ["data", "features", "labels", "metadata", "names"]:
                if n == "data":
                    self._data = self._data.sort_values("hash")
                    h = ["hash"] + sorted([c for c in self._data.columns if c not in ["hash", "label"]]) + ["label"]
                    self._data.to_csv(str(self.path.joinpath("data.csv")), sep=";", columns=h, index=False, header=True)
                else:
                    with self.path.joinpath(n + ".json").open('w+') as f:
                        json.dump(getattr(self, "_" + n), f, indent=2)
        else:
            self.logger.warning("No data to save")
    
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
    
    def fix(self, labels=None, detect=False, files=False, **kw):
        """ Make dataset's structure and files match. """
        self.backup = self
        labels = Dataset.labels(labels)
        # first, clean duplicates in data
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
                    self[self._names[h], True] = Detector.detect(e) if detect else l
        self._save()
    
    def is_valid(self):
        """ Check if this Dataset instance has a valid structure. """
        return Dataset.check(self.path)
    
    def list(self, **kw):
        """ List all the datasets from the given path. """
        print(mdv.main(Report(*Dataset.summarize(kw.get('path', "."), kw.get('show_all', False))).md()))
    
    def make(self, n=100, categories=["All"], balance=False, packer=None, refresh=False, **kw):
        """ Make n new samples in the current dataset among the given binary categories, balanced or not according to
             the number of distinct packers. """
        pbar = tqdm(total=n, unit="executable")
        self.backup = self
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
            p = sorted(list(set([l for l in self._data['label'].values if isinstance(l, str)])))
            self.logger.info("Used packers: %s" % ", ".join(p))
            self.logger.info("#Executables: %d" % l)
        self._save()
    
    def merge(self, path2=None, update=False, **kw):
        """ Merge another dataset with the current one ; precedence is set by the 'update' parameter. """
        self.backup = self
        ds2 = Dataset(path2)
        # add rows from the input dataset
        for row, name in ds2:
            self[name, update] = row
        # as the previous operation does not update categories and features, do it manually
        self._metadata.setdefault('categories', [])
        for category in ds2._metadata.get('categories', []):
            if category not in self._metadata['categories']:
                self._metadata['categories'].append(category)
        d = {k: v for k, v in (self if update else ds2)._features.items()}
        d.update((ds2 if update else self)._features)
        self._features = d
        # now copy files from the input dataset
        for f in ds2.files.listdir():
            Executable(f, dataset=self).copy()
        self._save()
    
    def remove(self, hash=None, start=None, end=None, label=None, **kw):
        """ Remove executables from the dataset given their hashes. """
        self.backup = self
        for h in (hash or []):
            del self[h]
        if start is not None or end is not None:
            remove = start is None
            for f in self.files.listdir(sort=True):
                h = f.filename
                if start == h:
                    remove = True
                if remove:
                    del self[h]
                if end == h:
                    remove = False
        if label is not None:
            for h, l in list(self._labels.items()):
                if label == l:
                    del self[h]
        self._save()
    
    def rename(self, path2=None, **kw):
        """ Rename the current dataset. """
        if not ts.Path(path2).exists():
            tmp = ts.TempPath(".dataset-backup", self._hash())
            self.path = self.path.rename(path2)
            tmp.rename(ts.TempPath(".dataset-backup", self._hash()))
    
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
        """ Revert to the latest version of the dataset (if a backup copy exists in /tmp. """
        b = self.backup
        if b is None:
            self.logger.warning("No backup found ; could not revert")
        else:
            self._remove()
            b._copy(self.path)
            b._remove()
            self._save()
    
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
    
    def update(self, source_dir=".", categories=["All"], labels=None, detect=False, **kw):
        """ Update the dataset with a folder of binaries, detecting used packers if 'detect' is set to True, otherwise
             packing randomly. If labels are provided, they are used instead of applying packer detection. """
        self.backup = self
        self.categories = categories
        labels = Dataset.labels(labels)
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
    def name(self):
        """ Get the name of the dataset composed with its list of categories. """
        return "%s(%s)" % (self.path.stem, ",".join(self.categories))
    
    @property
    def overview(self):
        """ Represent an overview of the dataset. """
        r = []
        categories = expand_categories("All")
        CAT = ["<20kB", "20-50kB", "50-100kB", "100-500kB", "500kB-1MB", ">1MB"]
        size_cat = lambda s: CAT[0] if s < 20 * 1024 else CAT[1] if 20 * 1024 <= s < 50 * 1024 else \
                             CAT[2] if 50 * 1024 <= s < 100 * 1024 else CAT[3] if 100 * 1024 <= s < 500 * 1024 else \
                             CAT[4] if 500 * 1024 <= s < 1024 * 1024 else CAT[5]
        data1, data2 = {}, {}
        total, totalp, d = 0, 0, {c: [0, 0] for c in CAT}
        if self.__per_category:
            for category in categories:
                d = {c: [0, 0] for c in CAT}
                for h, label in self._labels.items():
                    exe = Executable(self.files.joinpath(h), dataset=self)
                    if exe.category != category:
                        continue
                    s = size_cat(exe.size)
                    d[s][0] += 1
                    if label is not None:
                        d[s][1] += 1
                    data2.setdefault(category, [])
                    if len(data2[category]) < self.__limit:
                        row = [v if isinstance(v, str) else "" for k, v in self[h, True].items() \
                               if k in ["hash", "label"]]
                        row.insert(1, self._names[h])
                        data2[category].append(row)
                    elif len(data2[category]) == self.__limit:
                        data2[category].append(["...", "...", "..."])
                total, totalp = sum([v[0] for v in d.values()]), sum([v[1] for v in d.values()])
                if total == 0:
                    continue
                data1.setdefault(category, [])
                for c in CAT:
                    data1[category].append([c, d[c][0], "%.2f" % (100 * (float(d[c][0]) / total)) if total > 0 else 0,
                                            d[c][1], "%.2f" % (100 * (float(d[c][1]) / totalp)) if totalp > 0 else 0])
                data1[category].append(["Total", str(total), "", str(totalp), ""])
        else:
            d = {c: [0, 0] for c in CAT}
            for h, label in self._labels.items():
                exe = Executable(self.files.joinpath(h), dataset=self)
                s = size_cat(exe.size)
                d[s][0] += 1
                if label is not None:
                    d[s][1] += 1
                data2.setdefault('All', [])
                if len(data2['All']) < self.__limit:
                    row = [v if isinstance(v, str) else "" for k, v in self[h, True].items() if k in ["hash", "label"]]
                    row.insert(1, self._names[h])
                    data2['All'].append(row)
                elif len(data2['All']) == self.__limit:
                    data2['All'].append(["...", "...", "..."])
            total, totalp = sum([v[0] for v in d.values()]), sum([v[1] for v in d.values()])
            data1.setdefault('All', [])
            for c in CAT:
                data1['All'].append([c, d[c][0], "%.2f" % (100 * (float(d[c][0]) / total)) if total > 0 else 0,
                                        d[c][1], "%.2f" % (100 * (float(d[c][1]) / totalp)) if totalp > 0 else 0])
            data1['All'].append(["Total", str(total), "", str(totalp), ""])
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
                    r += [Table(data2[c], title=c, column_headers=["Hash", "Path", "Label"])]
                if c == "All":
                    break
        return r
    
    @staticmethod
    def check(folder):
        try:
            Dataset.validate(folder)
            return True
        except ValueError as e:
            return False
    
    @staticmethod
    def labels(labels):
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
        for dset in ts.Path(path or ".").listdir(Dataset.check):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                datasets.append([
                    dset.stem,
                    str(metadata['executables']),
                    ",".join(metadata['categories']),
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
    def validate(folder):
        f = ts.Path(folder)
        if not f.exists():
            raise ValueError("Folder does not exist")
        if not f.is_dir():
            raise ValueError("Input is not a folder")
        if not f.joinpath("files").exists():
            raise ValueError("Files subfolder does not exist")
        if not f.joinpath("files").is_dir():
            raise ValueError("Files subfolder is not a folder")
        for fn in ["data.csv", "features.json", "labels.json", "names.json"]:  # NB: metadata.json is optional
            if not f.joinpath(fn).exists():
                raise ValueError("Folder does not have %s" % fn)
        return f


class InconsistentDataset(ValueError):
    pass

