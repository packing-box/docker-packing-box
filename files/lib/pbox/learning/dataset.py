# -*- coding: UTF-8 -*-
from tinyscript.helpers import is_executable
from tqdm import tqdm

from .executable import Executable
from ..common.dataset import Dataset
from ..common.utils import backup


__all__ = ["open_dataset", "Dataset", "FilelessDataset"]


def open_dataset(folder):
    """ Open the target dataset with the right class. """
    if Dataset.check(folder):
        return Dataset(folder)
    elif FilelessDataset.check(folder):
        return FilelessDataset(folder)
    raise ValueError("Not a valid dataset")


class FilelessDataset(Dataset):
    """ Folder structure:
    
    [name]
      +-- data.csv          # metadata and labels of the executable
      +-- features.json     # dictionary of selected features and their descriptions
      +-- metadata.json     # simple statistics about the dataset
    """
    _files = False
    
    def _iter_with_features(self, feature=None, pattern=None):
        """ Convenience generator supplementing __iter__ for ensuring that feaures are also included. """
        if self._files:
            for exe in self.files.listdir(is_executable):
                exe = Executable(dataset=self, hash=exe.basename)
                exe.selection = feature or pattern
                self._features.update(exe.features)
                yield exe
        else:
            for exe in self._data.itertuples():
                exe = Executable(exe)
                exe.selection = feature or pattern
                yield exe
    Dataset._iter_with_features = _iter_with_features
    
    @backup
    def convert(self, feature=None, pattern=None, **kw):
        """ Convert a dataset with files to a dataset without files. """
        if not self._files:
            self.logger.warning("A fileless dataset cannot be converted to a dataset with files")
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
            exe = Executable(dataset=self, hash=h)
            exe.selection = feature or pattern
            self._features.update(exe.features)
            d = self[exe.hash, True]
            d.update(exe.data)
            self[exe.hash] = d
            pbar.update()
        self.files.remove(error=False)
        self._save()
        self.logger.debug("removing files...")
    Dataset.convert = convert

