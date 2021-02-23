# -*- coding: UTF-8 -*-
import pandas as pd
from tinyscript import b, colored, hashlib, json, logging, random, ts
from tinyscript.report import *
from tqdm import tqdm

from .__common__ import expand_categories
from .executable import Executable
from .packer import Packer


__all__ = ["Dataset", "PACKING_BOX_SOURCES"]


PACKING_BOX_SOURCES = ["/usr/bin", "/root/.wine/drive_c/windows", "/root/.wine32/drive_c/windows"]


class Dataset:
    """ File structure:
    
    [name]
      +-- files
      |     +-- {executables, renamed to their SHA256 hashes}
      +-- data.csv (contains the labels)
      +-- metadata.json
      +-- names.csv
    """
    @logging.bindLogger
    def __init__(self, destination_dir="dataset", source_dir=None, **kw):
        self.path = ts.Path(destination_dir, create=True)
        self.sources = source_dir or PACKING_BOX_SOURCES
        self.load()
    
    def __setattr__(self, name, value):
        if name == "categories":
            # get the list of packers related to the selected categories
            self._categories_exp = expand_categories(*value)
            self.packers = [p for p in Packer.registry if p.status > 2 and p.check(*self._categories_exp)]
            if len(self.packers) == 0:
                raise ValueError("No packer found for these categories")
        elif name == "sources":
            sources = []
            for src in value:
                src = ts.Path(src, expand=True)
                if not src.exists() or not src.is_dir():
                    continue
                sources.append(src)
            value = sources
        super(Dataset, self).__setattr__(name, value)
    
    def _add(self, executable, label=None, refresh=True):
        """ Add an executable based on its real name to the dataset. """
        e = executable
        if refresh:
            self._data = self._data[self._data['hash'] != e.hash]
        d = {'hash': h, 'label': label}
        d.update(e.data)
        self._data.insert(d, ignore_index=True)
        self._features.update(e.features)
        if not refresh:
            self._labels[e.hash] = label
            self._names[e.hash] = str(e)
    
    def _remove(self, executable):
        """ Remove an executable (by its real name or hash) from the dataset. """
        # first, ensure we handle the hash name (not the real one)
        h = {n: h for h, n in self._names.items()}.get(executable, executable)
        # then try removing
        for l in ["_labels", "_names"]:
            try:
                del getattr(self, l)[h]
            except KeyError:
                pass
        self._data = self._data[self._data.hash != h]
        self.path.joinpath("files", h).remove(error=False)
    
    def _walk(self):
        """ Walk the sources for random in-scope executables. """
        self.logger.info("Searching for executables...")
        m = 0
        candidates, hashes = [], []
        packers = [p.name for p in Packer.registry]
        for src in self.sources:
            for exe in ts.Path(src, expand=True).walk():
                exe = Executable(exe)
                exe.dataset = self
                if exe.category not in self._categories_exp:
                    continue
                if exe.filename in packers:
                    continue  # ignore packers themselves
                if exe.hash not in hashes:
                    hashes.append(exe.hash)
                    candidates.append(exe)
        random.shuffle(candidates)
        for c in candidates:
            yield c
    
    def fix(self, files=True):
        """ Make dataset's structure and files match. """
        if files:
            # fix wrt labels first
            for h, l in list(self._labels.items()):
                if not self.path.joinpath("files", h).exists():
                    del self._labels[h]
                    del self._data[h]
            for f in self.path.joinpath("files").listdir():
                if f.filename not in self._labels:
                    f.remove()
    
    def is_valid(self):
        """ Check if this Dataset instance has a valid structure. """
        return Dataset.check(self.path)
    
    def load(self):
        """ Load dataset's associated files or create them. """
        self.path.joinpath("files").mkdir(exist_ok=True)
        for n in ["data", "features", "labels", "metadata", "names"]:
            p = self.path.joinpath(n + [".json", ".csv"][n == "data"])
            if p.exists():
                if n == "data":
                    try:
                        self._data = pd.read_csv(str(p), sep=";").set_index('name')
                    except KeyError:
                        self._data = pd.DataFrame()
                else:
                    with p.open() as f:
                        setattr(self, "_" + n, json.load(f))
            else:
                setattr(self, "_" + n, {})
                p.write_text("{}")
        return self
    
    def make(self, n=100, categories=["All"], balance=False, packer=None, refresh=False, **kw):
        """ Make n new samples in the current dataset among the given binary categories, balanced or not according to
             the number of distinct packers. """
        pbar = tqdm(total=n, unit="executable")
        self.categories = categories
        self.logger.info("Source directories:    %s" % ",".join(map(str, self.sources)))
        self.logger.info("Considered categories: %s" % ",".join(categories))
        self.logger.info("Selected packers:      %s" % ",".join(["All"] if packer is None else \
                                                                [p.__class__.__name__ for p in packer]))
        self._metadata['categories'] = list(set(self._metadata.get('categories', []) + categories))
        # get executables to be randomly packed or not
        i, l = 0, self._data
        for exe in self._walk():
            if i >= n:
                break
            packers = [p for p in (packer or Packer.registry) if p in self.packers]
            if exe.destination.exists():
                # when refresh=True, the features are recomputed for the existing target executable ; it allows to
                #  recompute features for a previously made dataset if the list of features was updated
                if refresh:
                    self._add(exe, refresh=True)
                continue
            i += 1
            exe.copy()
            label = short_label = None
            if random.randint(0, len(packers) if balance else 1):
                if len(packers) == 0:
                    self.logger.critical("No packer left")
                    return
                random.shuffle(packers)
                for p in packers:
                    label = p.pack(str(dst.absolute()))
                    if not label or p._bad:
                        if label is False or p._bad:
                            self.logger.warning("Disabling %s..." % p.__class__.__name__)
                            self.packers.remove(p)
                            label = None
                        continue
                    else:  # consider short label (e.g. "midgetpack", not "midgetpack[<password>]")
                        short_label = label.split("[")[0]
                    break
            self._add(exe, short_label)
            pbar.update()
        if len(self._data) < l + n:
            self.logger.warning("Found too few candidate executables")
        # finally, save dataset's metadata and labels to JSON files
        self.save()
        self.logger.info("Used packers: %s" % ", ".join(sorted(list(self._data['label'].values))))
        return self
    
    def prepare(self):
        """ Compute and attach sets of data from self.data and self.target to the instance.
        
        :pre: the dataset is assumed to be balanced
        """
        self.logger.debug("> Preparing train and test subsets...")
        # scale the data
        self.data = MinMaxScaler().fit_transform(self.data)
        # prepare for sklearn
        class Dummy: pass
        self.train, self.test = Dummy(), Dummy()
        self.train.data, self.test.data, self.train.target, self.test.target = train_test_split(self.data, self.target)
        # prepare for Weka
        self.to_arff(WekaClassifier.train_file, self.train.data, self.train.target, self.features, self.name, self.labels)
        self.to_arff(WekaClassifier.test_file, self.test.data, self.test.target, self.features, self.name, self.labels)
        return self
    
    def reset(self):
        """ Truncate and recreate a blank dataset. """
        self.path.remove()
        self.load()
        return self
    
    def save(self):
        """ Save dataset's state to JSON files. """
        self._metadata['counts'] = self._data['label'].value_counts().to_dict()
        for n in ["data", "features", "labels", "metadata", "names"]:
            if n == "data":
                self._data.set_index('hash')
                headers = ["name"] + sorted([h for h in self._data.columns if h not in ["name", "label"]]) + ["label"]
                self._data.sort_index().to_csv(str(self.path.joinpath("data.csv")), sep=";", columns=headers)
            else:
                with self.path.joinpath(n + ".json").open('w+') as f:
                    json.dump(getattr(self, "_" + n), f, indent=2)
        return self
    
    @staticmethod
    def check(folder):
        try:
            Dataset.validate(folder)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate(folder):
        f = ts.Path(folder)
        if not f.exists():
            raise ValueError("Folder does not exist")
        if not f.joinpath("files").exists():
            raise ValueError("Files subfolder does not exist")
        if not f.joinpath("files").is_dir():
            raise ValueError("Files subfolder is not a folder")
        for fn in ["data.json", "features.json", "labels.json", "names.json"]:  # NB: metadata.json is optional
            if not f.joinpath(fn).exists():
                raise ValueError("Folder does not have %s" % fn)
        return f
    
    @classmethod
    def summary(cls, show=False):
        datasets = []
        headers = ["Name", "Size", "Packers"]
        for dset in ts.Path().listdir(Dataset.check):
            with dset.joinpath("metadata.json").open() as meta:
                metadata = json.load(meta)
            try:
                datasets.append([
                    dset.stem,
                    str(metadata['executables']),
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

