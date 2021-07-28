# -*- coding: UTF-8 -*-
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
        e = executable
        try:
            e, refresh = e
        except (TypeError, ValueError):
            refresh = False
        e = Executable(e, dataset=self)
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
            raise ValueError("A fileless dataset cannot be converted to a dataset with files")
        headers, fnames = self._data.columns[:-1], []  # columns[-1] is assumed to be "label"
        for exe in self.files.listdir(ts.is_executable):
            h = exe.basename
            exe = Executable(dataset=self, hash=h)
            exe.selection = features or pattern
            d = {'hash': h}
            d.update(exe.data)
            for n, v in exe.features.items():
                if n not in fnames:
                    fnames.append(n)
                self._data.at[self._data.hash == h, n] = v
        self._data = self._data[headers + sorted(fnames) + "label"]
        self._save()
    Dataset.convert = convert

