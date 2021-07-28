# -*- coding: UTF-8 -*-
from tinyscript.helpers import is_executable
from tqdm import tqdm

from .executable import Executable
from ..common.dataset import Dataset
from ..common.utils import backup


__all__ = ["Dataset", "FilelessDataset"]


class FilelessDataset(Dataset):
    """ Folder structure:
    
    [name]
      +-- data.csv          # metadata and labels of the executable
      +-- features.json     # dictionary of selected features and their descriptions
      +-- metadata.json     # simple statistics about the dataset
    """
    _files = False

    def __setitem__(self, executable, label):
        # compute the executable's metadata as normally
        super(FilelessDataset, self).__setitem__(executable, label)
        # then, compute features so that the file is no longer needed
        e = Executable(executable, dataset=self)
        if not hasattr(self, "_features"):
            self._features = {}
        self._features.update(e.features)
        # consider the case when 'label' is a dictionary with the executable's attribute, i.e. from another dataset
        d = {}
        if isinstance(label, dict):
            d = {k: v for k, v in label.items() if k in e.features}
        if not all(k in d for k in e.features):
            d.update(e.data)
        for n, v in d.items():
            self._data.at[self._data.hash == e.hash, n] = v
    
    @backup
    def convert(self, feature=None, pattern=None, **kw):
        """ Convert a dataset with files to a dataset without files. """
        if not self._files:
            self.logger.warning("A fileless dataset cannot be converted to a dataset with files")
            return
        self._files = False
        self.logger.debug("converting dataset...")
        headers, fnames = self._data.columns[:-1], []  # columns[-1] is assumed to be "label"
        self.path.joinpath("features.json").write_text("{}")
        self._features = {}
        pbar = tqdm(total=self._metadata['executables'], unit="executable")
        for exe in self.files.listdir(is_executable):
            h = exe.basename
            exe = Executable(dataset=self, hash=h)
            exe.selection = feature or pattern
            self[exe.hash] = self[exe.hash, True]
            pbar.update()
        self._data = self._data[headers.to_list() + sorted(fnames) + ["label"]]
        self.files.remove(error=False)
        self._save()
        self.logger.debug("removing files...")
    Dataset.convert = convert

