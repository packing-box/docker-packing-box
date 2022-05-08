# -*- coding: UTF-8 -*-
import matplotlib.pyplot
import plotext
from textwrap import wrap
from tinyscript import code, colored, itertools, ts
from tinyscript.helpers import ansi_seq_strip, get_terminal_size, ints2hex, is_executable, txt2bold, Path
from tqdm import tqdm

from .executable import Executable
from ..common.config import config
from ..common.dataset import Dataset
from ..common.utils import backup
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
    p = config['datasets'].joinpath(folder)
    if Dataset.check(folder):
        return Dataset(folder)
    if FilelessDataset.check(folder):
        return FilelessDataset(folder)
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
        l = self.logger
        if not self._files:
            l.warning("Already a fileless dataset")
            return
        l.info("Converting to fileless dataset...")
        l.info("Size of dataset:     %s" % ts.human_readable_size(self.path.size))
        if new_name is not None:
            ds = Dataset(new_name)
            ds.merge(self.path.basename, **kw)
            ds.convert(feature, pattern, **kw)
            l.info("Size of new dataset: %s" % ts.human_readable_size(ds.path.size))
            return
        self._files = False
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
            self[exe.hash] = (d, True)  # True: force updating the row
            pbar.update()
        pbar.close()
        l.debug("removing files...")
        self.files.remove(error=False)
        l.debug("removing eventual backups...")
        try:
            self.backup.purge()
        except AttributeError:
            pass
        self._save()
        l.info("New size of dataset: %s" % ts.human_readable_size(self.path.size))
    Dataset.convert = convert
    
    def features(self, feature, output_format=None, dpi=200, multiclass=False, **kw):
        """ Plot the distribution of the given feature. """
        l = self.logger
        if not isinstance(feature, (tuple, list)):
            feature = [feature]
        l.info("Counting values for feature%s %s..." % (["", "s"][len(feature) > 1], ", ".join(feature)))
        # start counting, keeping 'Not packed' counts separate (to prevent it from being sorted with others)
        counts_np, counts, labels = {}, {}, []
        for exe in self._iter_with_features(feature):
            v = tuple(exe.data[f] for f in feature)
            counts_np.setdefault(v, 0)
            counts.setdefault(v, {} if multiclass else {'Packed': 0})
            lbl = str(exe.label)
            if lbl == "nan":
                counts_np[v] += 1
            elif multiclass:
                lbl = Packer.get(lbl).cname
                counts[v].setdefault(lbl, 0)
                counts[v][lbl] += 1
                if lbl not in labels:
                    labels.append(lbl)
            else:
                counts[v]['Packed'] += 1
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
        l.debug("Plotting...")
        width = get_terminal_size()[0] if output_format is None else 60
        plt = plotext if output_format is None else matplotlib.pyplot
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
        values = [[] for i in range(len(counts[list(counts.keys())[0]]))]  # series per label (Not packed, Amber, ...)
        for v in counts.values():
            for i, sv in enumerate(v.values()):
                values[i].append(sv)
        percentages = [[100 * x / total for x in l] for l in values]
        # set color palette
        cmap = ["green"] + [list(COLORMAP.keys())[i % len(COLORMAP)] for i in range(len(values) - 1)]
        labels = list(counts[list(counts.keys())[0]].keys())
        # display plot
        plur = ["", "s"][len(feature) > 1]
        x_label, y_label = "Percentages of samples for the selected feature%s" % plur, "Feature value%s" % plur
        if output_format is None:
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
            img_name = ["", "combo-"][len(feature) > 1] + feature[0] + "." + output_format
            l.debug("saving image to %s..." % img_name)
            plt.savefig(img_name, img_format=output_format, dpi=dpi, bbox_inches="tight")
    Dataset.features = features
    
    @backup
    def merge(self, name2=None, **kw):
        """ Merge another dataset with the current one. """
        l = self.logger
        ds2 = Dataset(name2) if Dataset.check(name2) else FilelessDataset(name2)
        cls1, cls2 = self.__class__.__name__, ds2.__class__.__name__
        if cls1 != cls2:
            l.error("Cannot merge %s and %s" % (cls1, cls2))
            return
        # add rows from the input dataset
        l.info("Merging rows from %s into %s..." % (ds2.basename, self.basename))
        pbar = tqdm(total=ds2._metadata['executables'], unit="executable")
        for r in ds2:
            self[Executable(hash=r.hash, dataset=ds2, dataset2=self)] = r._row._asdict()
            pbar.update()
        pbar.close()
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

