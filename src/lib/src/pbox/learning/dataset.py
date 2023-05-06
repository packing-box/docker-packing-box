# -*- coding: UTF-8 -*-
from tinyscript import random
from tinyscript.helpers import human_readable_size, Path
from tinyscript.report import *

from .executable import Executable
from .pipeline import *
from .plot import *
from ..common.config import *
from ..common.dataset import Dataset
from ..common.rendering import progress_bar
from ..common.utils import *


__all__ = ["open_dataset", "Dataset", "FilelessDataset"]


def open_dataset(folder, **kw):
    """ Open the target dataset with the right class. """
    for cls in [Dataset, FilelessDataset]:
        try:
            return cls(cls.validate(folder, **kw), **kw)
        except ValueError:
            pass
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
        d = self[exe.hash, True]  # retrieve executable's record as a dictionary
        d.update(exe.data)        # be sure to include the features
        return d
    Dataset._compute_features = _compute_features
    
    def _compute_all_features(self):
        """ Convenience function for computing the self._data pandas.DataFrame containing the feature values. """
        self.logger.info("Computing features..." if self._files else "Loading features...")
        with progress_bar() as p:
            for exe in p.track(self):
                d = self._compute_features(exe)
                if self._files:
                    self[exe.hash] = (d, True)  # True: force updating the row
    Dataset._compute_all_features = _compute_all_features
    
    def browse(self, query=None, no_feature=False, **kw):
        if not no_feature:
            self._compute_all_features()
        with data_to_temp_file(filter_data(self._data, query, logger=self.logger), prefix="dataset-browsing-") as tmp:
            edit_file(tmp, logger=self.logger)
    Dataset.browse = browse
    
    @backup
    def convert(self, new_name=None, silent=False, **kw):
        """ Convert a dataset with files to a dataset without files. """
        l = self.logger
        l_info = getattr(l, ["info", "debug"][silent])
        if not self._files:
            l.warning("Already a fileless dataset")
            return
        if new_name is not None:
            ds = Dataset(new_name, **kw)
            ds.merge(self.path.basename, silent=silent, **kw)
            ds.convert(silent=silent, **kw)
            return
        l_info("Converting to fileless dataset...")
        s1 = self.path.size
        l_info("Size of dataset:     %s" % human_readable_size(s1))
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
        l_info("Size of new dataset: %s (compression factor: %d)" % (human_readable_size(s2), int(s1/s2)))
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
            with progress_bar("packed samples") as p:
                for exe in enumerate(lst[:n]):
                    fn = "%s_%s" % (exe.label, Path(exe.realpath).filename)
                    if fn in tmp:
                        l.warning("duplicate '%s'" % fn)
                        continue
                    exe.destination.copy(dst.joinpath(fn))
                    tmp.append(fn)
            return
        if self._files:
            l.info("Computing features...")
            self._compute_all_features()
        fields = ["hash"] + Executable.FIELDS
        fnames = [h for h in self._data.columns if h not in fields + ["label", "Index"]]
        c = fields[:-1] + fnames + ["label"]
        d = self._data[c].values.tolist()
        d.insert(0, c)
        ext = ".%s" % format
        if not dst.endswith(ext):
            dst += ext
        from dsff import DSFF
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
        l = self.logger
        l_info = getattr(l, ["info", "debug"][silent])
        ds2 = open_dataset(name2, name_check=False)
        if new_name is not None:
            dstype = FilelessDataset if not self._files or not ds2._files else Dataset
            ds = dstype(new_name)
            if dstype is FilelessDataset and self._files:
                temp_name = random.randstr(16)
                self.convert(temp_name, check=False)
                ds.merge(temp_name)
                FilelessDataset(temp_name, check=False).purge()
            else:
                ds.merge(self.path.basename)
            if dstype is FilelessDataset and ds2._files:
                temp_name = random.randstr(16)
                ds2.convert(temp_name, check=False)
                ds.merge(temp_name)
                FilelessDataset(temp_name, check=False).purge()
            else:
                ds.merge(name2)
            return
        if self.__class__ is Dataset and ds2.__class__ is FilelessDataset:
            l.error("Cannot merge a fileless dataset into a dataset (because files are missing)")
            return
        # add rows from the input dataset
        l_info("Merging rows from %s into %s..." % (ds2.basename, self.basename))
        with progress_bar(silent=silent) as p:
            for r in p.track(ds2):
                self[Executable(hash=r.hash, dataset=ds2, dataset2=self)] = r._row._asdict()
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
    
    def preprocess(self, query=None, preprocessor=None, **kw):
        """ Preprocess dataset given selected features and preprocessors and display it with visidata for review. """
        self._compute_all_features()
        result = pd.DataFrame()
        for col in ["hash"] + Executable.FIELDS:
            result[col] = self._data[col]
        fnames = sorted(self._features.keys())
        data = self._data[fnames]
        if preprocessor:
            p = DebugPipeline()
            make_pipeline(p, preprocessor, self.logger)
            p.steps.append(("debug", DebugTransformer()))
            df = pd.DataFrame(p.preprocess(np.array(data)), columns=data.columns)
        for col in data.columns:
            if preprocessor:
                result[col] = df[col]
                col2 = "*" + col
                result[col2] = data[col]
            else:
                result[col] = data[col]
        result['label'] = self._data['label']
        with data_to_temp_file(filter_data(result, query, logger=self.logger), prefix="dataset-preproc-") as tmp:
            edit_file(tmp, logger=self.logger)
    Dataset.browse = browse
    
    @staticmethod
    def count():
        return sum(1 for _ in Path(config['datasets']).listdir(Dataset.check or FilelessDataset.check))
    Dataset.count = count
    
    @staticmethod
    def iteritems(instantiate=False):
        for dataset in Path(config['datasets']).listdir(lambda f: Dataset.check(f) or FilelessDataset.check(f)):
            yield open_dataset(dataset) if instantiate else dataset
    Dataset.iteritems = iteritems
    
    @staticmethod
    def open(folder, **kw):
        return open_dataset(folder, **kw)
    Dataset.open = open
    
    @staticmethod
    def summarize(show=False, hide_files=False):
        _, table = Dataset.summarize(show, hide_files)
        _, table2 = Dataset.summarize(show, hide_files, FilelessDataset.check)
        t, t2 = [] if table is None else table.data, [] if table2 is None else table2.data
        datasets = sorted(t + t2, key=lambda x: x[0])
        if len(datasets) > 0:
            table = Table(datasets, column_headers=(table or table2).column_headers)
            return [Section("Datasets (%d)" % len(table.data)), table]
        return None, None

