# -*- coding: UTF-8 -*-
from tinyscript.helpers import is_executable
from tqdm import tqdm

from .executable import Executable
from ..common.config import config
from ..common.dataset import Dataset
from ..common.utils import backup


__all__ = ["open_dataset", "Dataset", "FilelessDataset"]


def open_dataset(folder):
    """ Open the target dataset with the right class. """
    p = config['datasets'].joinpath(folder)
    if Dataset.check(folder):
        return Dataset(folder)
    if FilelessDataset.check(folder):
        return Dataset(folder)
    elif Dataset.check(p):
        return Dataset(p)
    elif FilelessDataset.check(p):
        return FilelessDataset(p)
    raise ValueError("%s is not a valid dataset" % folder)


class FilelessDataset(Dataset):
    """ Folder structure:
    
    [name]
      +-- data.csv          # metadata and labels of the executable
      +-- features.json     # dictionary of selected features and their descriptions
      +-- metadata.json     # simple statistics about the dataset
    """
    _files = False
    
    def __iter__(self):
        """ Iterate over the dataset. """
        for row in self._data.itertuples():
            e = Executable(row, dataset=self)
            e._row = row
            yield e
    Dataset.__iter__ = __iter__
    
    def _iter_with_features(self, feature=None, pattern=None):
        """ Convenience generator supplementing __iter__ for ensuring that feaures are also included. """
        if self._files:
            for exe in self:
                exe.selection = feature or pattern
                if not hasattr(self, "_features"):
                    self._features = {}
                self._features.update(exe.features)
                yield exe
        else:
            for exe in self:
                exe.selection = feature or pattern
                yield exe
    Dataset._iter_with_features = _iter_with_features
    
    @backup
    def convert(self, feature=None, pattern=None, new_name=None, **kw):
        """ Convert a dataset with files to a dataset without files. """
        if not self._files:
            self.logger.warning("A fileless dataset cannot be converted to a dataset with files")
            return
        if new_name is not None:
            ds = Dataset(new_name)
            ds.merge(self.path.basename, **kw)
            ds.convert(feature, pattern, **kw)
            return
        self._files = False
        self.logger.debug("converting dataset...")
        self.path.joinpath("features.json").write_text("{}")
        self._features = {}
        pbar = tqdm(total=self._metadata['executables'], unit="executable")
        if not hasattr(self, "_features"):
            self._features = {}
        for exe in self._iter_with_features(feature, pattern):
            h = exe.basename
            self._features.update(exe.features)
            d = self[exe.hash, True]
            d.update(exe.data)
            self[exe.hash] = d
            pbar.update()
        self.logger.debug("removing files...")
        self.backup.purge()
        self.files.remove(error=False)
        self._save()
    Dataset.convert = convert
    
    @backup
    def merge(self, name2=None, **kw):
        """ Merge another dataset with the current one. """
        ds2 = Dataset(name2) if Dataset.check(name2) else FilelessDataset(name2)
        cls1, cls2 = self.__class__.__name__, ds2.__class__.__name__
        if cls1 != cls2:
            self.logger.error("Cannot merge %s and %s" % (cls1, cls2))
            return
        # add rows from the input dataset
        self.logger.debug("merging rows from %s..." % ds2.path)
        pbar = tqdm(total=ds2._metadata['executables'], unit="executable")
        for r in ds2:
            self[Executable(hash=r.hash, dataset=ds2, dataset2=self)] = r._row._asdict()
            pbar.update()
        # as the previous operation does not update categories and features, do it manually
        self._metadata.setdefault('categories', [])
        for category in ds2._metadata.get('categories', []):
            if category not in self._metadata['categories']:
                self._metadata['categories'].append(category)
        self._metadata['counts'] = self._data.label.value_counts().to_dict()
        self._metadata['executables'] = len(self)
        self._metadata.setdefault('sources', [])
        if str(ds2.path) not in self._metadata['sources']:
            self._metadata['sources'].append(str(ds2.path))
        if hasattr(self, "_features") and hasattr(ds2, "_features"):
            d = {k: v for k, v in ds2._features.items()}
            d.update(self._features)
            self._features = d
        self._save()
    Dataset.merge = merge

