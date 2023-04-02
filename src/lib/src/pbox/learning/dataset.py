# -*- coding: UTF-8 -*-
from dsff import DSFF
from tinyscript.helpers import human_readable_size, Path
from tinyscript.report import *

from .executable import Executable
from .plot import *
from ..common.config import *
from ..common.dataset import Dataset
from ..common.utils import *


__all__ = ["open_dataset", "Dataset", "FilelessDataset"]


def open_dataset(folder):
    """ Open the target dataset with the right class. """
    if Dataset.check(folder):
        return Dataset(folder)
    if FilelessDataset.check(folder):
        return FilelessDataset(folder)
    p = config['datasets'].joinpath(folder)
    if Dataset.check(p):
        return Dataset(p)
    if FilelessDataset.check(p):
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
        """ Iterate over dataset's sample executables, computing the features and filling their dictionary too. """
        if not hasattr(self, "_features"):
            self._features = {}
        for row in self._data.itertuples():
            exe = Executable(row, dataset=self)
            exe._row = row
            self._features.update(exe.features)
            yield exe
    Dataset.__iter__ = __iter__
    
    def _compute_features(self, exe):
        """ Compute the features for a single Executable instance. """
        exe = Executable(exe, dataset=self, force=True)
        d = self[exe.hash, True]
        d.update(exe.data)
        return d
    Dataset._compute_features = _compute_features
    
    def _compute_all_features(self):
        """ Convenience function for computing the self._data pandas.DataFrame containing the feature values. """
        l = self.logger
        if self._files:
            l.info("Computing features...")
            pbar = tqdm(total=self._metadata['executables'], unit="executable")
        else:
            l.info("Loading features...")
        for exe in self:
            d = self[exe.hash, True]  # retrieve executable's record as a dictionary
            d.update(exe.data)        # be sure to include the features
            if self._files:
                self[exe.hash] = (d, True)  # True: force updating the row
                pbar.update()
        if self._files:
            pbar.close()
    Dataset._compute_all_features = _compute_all_features
    
    def browse(self, query=None, **kw):
        self._compute_all_features()
        with data_to_temp_file(filter_data(self._data, query, logger=self.logger), prefix="dataset-features-") as tmp:
            edit_file(tmp, logger=self.logger)
    Dataset.browse = browse
    
    @backup
    def convert(self, new_name=None, **kw):
        """ Convert a dataset with files to a dataset without files. """
        l = self.logger
        if not self._files:
            l.warning("Already a fileless dataset")
            return
        if new_name is not None:
            ds = Dataset(new_name)
            ds.merge(self.path.basename, silent=True, **kw)
            ds.convert(**kw)
            return
        l.info("Converting to fileless dataset...")
        s1 = self.path.size
        l.info("Size of dataset:     %s" % human_readable_size(s1))
        self.path.joinpath("features.json").write_text("{}")
        self._compute_all_features()
        l.debug("removing files...")
        self.files.remove(error=False)
        self._files = False
        l.debug("removing eventual backups...")
        try:
            self.backup.purge()
        except AttributeError:
            pass
        self._save()
        s2 = self.path.size
        l.info("Size of new dataset: %s (compression factor: %d)" % (human_readable_size(s2), int(s1/s2)))
    Dataset.convert = convert
    
    def export(self, format=None, output=None, **kw):
        """ Export either packed executables from the dataset to the given output folder or the complete dataset to a
             given format. """
        l = self.logger
        dst = output or self.basename
        if format == "packed-samples":
            if not self._files:
                l.warning("Packed samples can only be exported from a normal dataset (not on a fileless one)")
                return
            dst, n = Path(dst, create=True), kw.get('n', 0)
            lst, tmp = [e for e in self if e.label not in [NOT_PACKED, NOT_LABELLED]], []
            if n > len(lst):
                l.warning("%d packed samples were requested but only %d were found" % (n, len(lst)))
            n = min(n, len(lst))
            l.info("Exporting %d packed executables from %s to '%s'..." % (n, self.basename, dst))
            if 0 < n < len(lst):
                random.shuffle(lst)
            pbar = tqdm(total=n or len(lst), unit="packed executable")
            for i, exe in enumerate(lst):
                if i >= n:
                    break
                fn = "%s_%s" % (exe.label, Path(exe.realpath).filename)
                if fn in tmp:
                    l.warning("duplicate '%s'" % fn)
                    n += 1
                    continue
                exe.destination.copy(dst.joinpath(fn))
                tmp.append(fn)
                pbar.update()
            pbar.close()
            return
        if self._files:
            l.info("Computing features...")
            self._compute_all_features()
        try:
            self._metadata['counts'] = self._data.label.value_counts().to_dict()
        except AttributeError:
            self.logger.warning("No label found")
            return
        self._metadata['executables'] = len(self)
        self._metadata['formats'] = sorted(collapse_formats(*self._metadata['formats']))
        self._data = self._data.sort_values("hash")
        fields = ["hash"] + Executable.FIELDS + ["label"]
        fnames = [h for h in self._data.columns if h not in fields + ["Index"]]
        c = fields[:-1] + fnames + [fields[-1]]
        d = self._data[c].values.tolist()
        d.insert(0, c)
        ext = ".%s" % format
        if not dst.endswith(ext):
            dst += ext
        if format == "dsff":
            l.info("Exporting dataset %s to '%s'..." % (self.basename, dst))
            with DSFF(self.basename, 'w+') as f:
                f.write(d, self._features, self._metadata)
            Path(self.basename + ext).rename(dst)
        elif format in ["arff", "csv"]:
            l.info("Exporting dataset %s to '%s'..." % (self.basename, dst))
            with DSFF("<memory>") as f:
                f.name = self.basename
                f.write(d, self._features, self._metadata)
                getattr(f, "to_%s" % format)()
            Path(self.basename + ext).rename(dst)
        else:
            raise ValueError("Unknown target format (%s)" % format)
    Dataset.export = export
    
    @backup
    def merge(self, name2=None, new_name=None, silent=False, **kw):
        """ Merge another dataset with the current one. """
        if new_name is not None:
            ds = type(self)(new_name)
            ds.merge(self.path.basename)
            ds.merge(name2)
            ds.path.joinpath("files").remove(False)
            return
        l = self.logger
        ds2 = Dataset(name2) if Dataset.check(name2) else FilelessDataset(name2)
        cls1, cls2 = self.__class__.__name__, ds2.__class__.__name__
        if cls1 != cls2:
            l.error("Cannot merge %s and %s" % (cls1, cls2))
            return
        # add rows from the input dataset
        getattr(l, ["info", "debug"][silent])("Merging rows from %s into %s..." % (ds2.basename, self.basename))
        if not silent:
            pbar = tqdm(total=ds2._metadata['executables'], unit="executable")
        for r in ds2:
            self[Executable(hash=r.hash, dataset=ds2, dataset2=self)] = r._row._asdict()
            if not silent:
                pbar.update()
        if not silent:
            pbar.close()
        # as the previous operation does not update formats and features, do it manually
        self._metadata.setdefault('formats', [])
        for fmt in ds2._metadata.get('formats', []):
            if fmt not in self._metadata['formats']:
                self._metadata['formats'].append(fmt)
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
    
    def plot(self, subcommand=None, **kw):
        """ Plot something about the dataset. """
        plot(self, "ds-%s" % subcommand, **kw)
    Dataset.plot = plot
    
    @staticmethod
    def count():
        return sum(1 for _ in Path(config['datasets']).listdir(Dataset.check or FilelessDataset.check))
    Dataset.count = count
    
    @staticmethod
    def iteritems(instantiate=False):
        for dataset in Path(config['datasets']).listdir(Dataset.check):
            yield open_dataset(dataset) if instantiate else dataset
    Dataset.iteritems = iteritems
    
    @staticmethod
    def summarize(path=None, show=False, hide_files=False):
        _, table = Dataset.summarize(path, show, hide_files)
        _, table2 = Dataset.summarize(path, show, hide_files, FilelessDataset.check)
        t, t2 = [] if table is None else table.data, [] if table2 is None else table2.data
        datasets = sorted(t + t2, key=lambda x: x[0])
        if len(datasets) > 0:
            table = Table(datasets, column_headers=(table or table2).column_headers)
            return [Section("Datasets (%d)" % len(table.data)), table]
        return None, None

