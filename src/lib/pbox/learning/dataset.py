# -*- coding: UTF-8 -*-
import matplotlib.pyplot
import pandas
import plotext
from dsff import DSFF
from sklearn.covariance import empirical_covariance
from sklearn.preprocessing import StandardScaler
from textwrap import wrap
from tinyscript import code, colored, itertools, ts
from tinyscript.helpers import ansi_seq_strip, get_terminal_size, ints2hex, txt2bold, Path
from tqdm import tqdm

from .executable import Executable
from ..common.config import *
from ..common.dataset import Dataset
from ..common.utils import *
from ..items.packer import Packer


__all__ = ["open_dataset", "Dataset", "FilelessDataset"]


COLORMAP = {
    'red':        (255, 0,   0),
    'lightCoral': (240, 128, 128),
    'purple':     (128, 0,   128),
    'peru':       (205, 133, 63),
    'salmon':     (250, 128, 114),
    'rosyBrown':  (188, 143, 143),
    'sandyBrown': (244, 164, 96),
    'sienna':     (160, 82,  45),
    'plum':       (221, 160, 221),
    'pink':       (255, 192, 203),
    'tan':        (210, 180, 140),
    'tomato':     (255, 99,  71),
    'violet':     (238, 130, 238),
    'magenta':    (255, 0,   255),
    'fireBrick':  (178, 34,  34),
    'indigo':     (75,  0,   130),
}


# patch plotext to support Y labels with ANSI sequences for colored text
code.add_line(plotext._monitor.monitor_class.build_plot, 1, "from tinyscript.helpers import ansi_seq_strip")
code.replace(plotext._monitor.monitor_class.build_plot, "len(el[0])", "len(ansi_seq_strip(el[0]))")

matplotlib.pyplot.rcParams['font.family'] = "serif"


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
    
    def _compute_features(self):
        """ Convenience funcion for computing the self._data pandas.DataFrame containing the feature values. """
        pbar = tqdm(total=self._metadata['executables'], unit="executable")
        for exe in self:
            h = exe.basename
            d = self[exe.hash, True]
            d.update(exe.data)
            self[exe.hash] = (d, True)  # True: force updating the row
            pbar.update()
        pbar.close()
    Dataset._compute_features = _compute_features
    
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
        l.info("Size of dataset:     %s" % ts.human_readable_size(s1))
        self._files = False
        self.path.joinpath("features.json").write_text("{}")
        self._compute_features()
        l.debug("removing files...")
        self.files.remove(error=False)
        l.debug("removing eventual backups...")
        try:
            self.backup.purge()
        except AttributeError:
            pass
        self._save()
        s2 = self.path.size
        l.info("Size of new dataset: %s (compression factor: %d)" % (ts.human_readable_size(s2), int(s1/s2)))
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
            self._compute_features()
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
    
    def features(self, **kw):
        self._compute_features()
        with data_to_temp_file(self._data, prefix="dataset-features-") as tmp:
            edit_file(tmp, logger=self.logger)
    Dataset.features = features
    
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
    
    def plot(self, feature, format=None, dpi=200, multiclass=False, **kw):
        """ Plot the distribution of the given feature or multiple features combined. """
        l = self.logger
        if not isinstance(feature, (tuple, list)):
            feature = [feature]
        l.info("Counting values for feature%s %s..." % (["", "s"][len(feature) > 1], ", ".join(feature)))
        # start counting, keeping 'Not packed' counts separate (to prevent it from being sorted with others)
        counts_np, counts, labels, data = {}, {}, [], pandas.DataFrame()
        for exe in self:
            row = {f: v for f, v in exe.data.items() if f in feature}
            data = data.append(row, ignore_index=True)
            v = tuple(row.values())
            counts_np.setdefault(v, 0)
            counts.setdefault(v, {} if multiclass else {'Packed': 0})
            lbl = str(exe.label)
            if lbl == NOT_PACKED:
                counts_np[v] += 1
            elif multiclass:
                lbl = Packer.get(lbl).cname
                counts[v].setdefault(lbl, 0)
                counts[v][lbl] += 1
                if lbl not in labels:
                    labels.append(lbl)
            else:
                counts[v]['Packed'] += 1
        data = StandardScaler().fit_transform(data)
        # compute variance and covariance (if multiple features)
        cov_matrix = empirical_covariance(data)
        if len(feature) > 1:
            var = "Variances:\n- " + "\n- ".join("%s: %.03f" % (f, cov_matrix[i][i]) for i, f in enumerate(feature))
            covar = "Covariances:\n"
            for i in range(len(cov_matrix)):
                for j in range(i + 1, len(cov_matrix)):
                    covar += "- %s / %s: %.03f\n" % (feature[i], feature[j], cov_matrix[i][j])
        else:
            var = "Variance: %.03f" % cov_matrix[0][0]
        # be sure to have values for every label (it was indeed not seen if 0, so set the default value)
        for v, d in counts.items():
            if multiclass:
                for lbl in labels:
                    d.setdefault(lbl, 0)
            else:
                d.setdefault('Packed', 0)
        l.debug("sorting feature values...")
        # sort counts by feature value and by label
        counts = {k: {sk: sv for sk, sv in sorted(v.items(), key=lambda x: x[0].lower())} \
                  for k, v in sorted(counts.items(), key=lambda x: x[0])}
        # merge counts of not packed and other counts
        all_counts = {k: {'Not packed': v} for k, v in sorted(counts_np.items(), key=lambda x: x[0])}
        for k, v in counts.items():
            for sk, sv in v.items():
                all_counts[k][sk] = sv  # force keys order
        counts = all_counts
        l.debug("reformatting feature values...")
        vtype = str
        #  transform {0,1} to False|True
        if set(counts.keys()) == {0., 1.}:
            counts = {k == 1.: v for k, v in counts.items()}
            vtype = bool
        #  e.g. aggregate (141, 85) in its hexstring '8d55'
        elif all(f.startswith("byte_") for f in feature):
            counts = {ints2hex(*tuple(int(sk) for sk in k)): v for k, v in counts.items()}
            vtype = hex
        #  convert floats to integers if no decimal present
        elif all(all(int(sk) == sk for sk in k) for k in counts.keys()):
            counts = {tuple(int(sk) for sk in k): v for k, v in counts.items()}
            vtype = int
        l.debug("plotting...")
        width = get_terminal_size()[0] if format is None else 60
        plt = plotext if format is None else matplotlib.pyplot
        try:
            title = self._features[feature[0]] if len(feature) == 1 else \
                    "\n".join(wrap("combination of %s" % ", ".join(self._features[f] for f in feature), width))
            title = title[0].upper() + title[1:] + " for dataset %s" % self.name
        except KeyError as e:
            l.error("Feature '%s' does not exist in the target dataset." % e.args[0])
            l.warning("This may occur when this feature was renamed in pbox.learning.features with a newer version.")
            return
        # compute percentages
        total = sum(sum(x.values()) for x in counts.values())
        values = [[] for i in range(len(counts[next(iter(counts))]))]  # series per label (Not packed, Amber, ...)
        for v in counts.values():
            for i, sv in enumerate(v.values()):
                values[i].append(sv)
        percentages = [[100 * x / total for x in l] for l in values]
        # set color palette
        cmap = ["green"] + [list(COLORMAP.keys())[i % len(COLORMAP)] for i in range(len(values) - 1)]
        labels = list(counts[next(iter(counts))].keys())
        # display plot
        plur = ["", "s"][len(feature) > 1]
        x_label, y_label = "Percentages of samples for the selected feature%s" % plur, "Feature value%s" % plur
        if format is None:
            pcmap = [(40, 210, 40)] + [list(COLORMAP.values())[i % len(COLORMAP)] for i in range(len(values) - 1)]
            # uses valid colors as defined in plotext
            yticks = [(str(k[0]) if isinstance(k, (tuple, list)) and len(k) == 1 else str(k),
                       "(%s)" % "|".join([colored(str(sv), c) for sv, c in zip(v.values(), cmap)])) \
                      for k, v in counts.items()]
            lmax = [max(map(len, [t for t, _ in yticks])), max(map(len, [t for _, t in yticks]))]
            yticks = ["%s %s" % (t1.ljust(lmax[0]), t2.rjust(lmax[1])) for t1, t2 in yticks]
            plt.stacked_bar(yticks, percentages, color=pcmap, orientation="h", marker="sd", minimum=.0, width=.1)
            # print title separately to put newlines around it (plotext strips newlines)
            print("\n%s\n" % "\n".join(txt2bold(l.center(width)) for l in title.splitlines()))
            plt.clc()
            plt.plotsize(width, 2 * (len(counts) + 1))
            plt.show()
            print(y_label + x_label.center(width - len(y_label)))
            # manually make the legend
            leg = (8 * " ").join(colored("██ %s" % n, c) for n, c in zip(labels, cmap))
            print("\n" + leg.center(width + (len(leg) - len(ansi_seq_strip(leg)))))
            print("\n" + var)
            if len(feature) > 1:
                print("\n" + covar)
        else:
            font = {'color': "lightgray", 'size': 10}
            yticks = [str(k[0]) if isinstance(k, (tuple, list)) and len(k) == 1 else str(k) \
                      for k in counts.keys()]
            plt.figure(figsize=(8, (len(title.splitlines()) * 24 + 11 * len(counts) + 120) / 80))
            plt.title(title, pad=20)
            plt.xlabel(x_label, fontdict=font)
            plt.ylabel(y_label, fontdict=font)
            starts = [0 for i in range(len(values[0]))]
            for p, lb ,c, v in zip(percentages, labels, cmap, values):
                b = plt.barh(yticks, p, label=lb, color=c, left=starts)
                starts = [x + y for x, y in zip(starts, p)]
                plt.bar_label(b, labels=["" if x == 0 else x for x in v], label_type="center", color="white")
            plt.yticks(**({'family': "monospace", 'fontsize': 8} if vtype is hex else {'fontsize': 9}))
            plt.legend()
            img_name = ["", "combo-"][len(feature) > 1] + feature[0] + "." + format
            l.debug("saving image to %s..." % img_name)
            plt.savefig(img_name, format=format, dpi=dpi, bbox_inches="tight")
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

