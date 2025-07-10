# -*- coding: UTF-8 -*-
"""
Assess the overall quality of the dataset by computing a normalized score based on multiple individual metrics.
"""
from ...helpers import *


__all__ = ["balance", "Scores"]

_WEIGHTS = {
    'completeness':  .5,
    'uniqueness':    .5,
    'similarity':    1.,
    'label_balance': 1.,
    'portability':   1.,
    'file_balance':  1.,
    'consistency':   .5,
    'outliers':      1.,
}


_isnumber = lambda t: t in (np.float16, np.float32, np.float64, np.float128, np.int8, np.int16, np.int32, np.int64)


def balance(dataset, field):
    try:
        data = pd.Series([getattr(exe, field) for exe in dataset]) if dataset._files else dataset._data.get(field)
    except AttributeError:
        dataset.logger.warning(f"field {field} does not exist")
        return
    return min(data.std() / m if (m := data.mean()) > 0 else 0. / 2, 1.) if _isnumber(data.dtype) else \
           1. - abs((counts := data.value_counts(normalize=True)).max() - counts.min())


class Scores:
    def __init__(self, dataset, file_balance_fields=("format", "signature", "size"), similarity_threshold=1., **kw):
        self._ds = dataset
        self._log = dataset.logger
        self.__fbf = file_balance_fields
        self.__st = similarity_threshold
        self.__w = _WEIGHTS
    
    @save_figure
    def plot(self, **kw):
        """ Plot a radar chart with the conigured metrics. """
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.lines import Line2D
        # custom color depending on the [0., 1.] scale of the overall score with a colormap from red to green
        color = LinearSegmentedColormap.from_list("score", ["red", "yellow", "green"])(self.overall)
        # format data according to individual scores and weighted scores (as they contribute to the overall score)
        data = [{m: getattr(self, m) for m in self.__w.keys() if getattr(self, m) is not None},
                {m: getattr(self, m) * w for m, w in self.__w.items() if getattr(self, m) is not None}]
        # create the radar chart, starting from the north and clockwise
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'polar': True})
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        # plot the title (aka suptitle in the figure object)
        t = fig.suptitle(kw.get('title') or f"Quality of dataset {self._ds.basename}", **kw['title-font'])
        # then plot the subtitle with Text objects, computing positions accordingly
        st_prefix, st_score = "Overall score: ", f"{self.overall:.3f}"
        h_pixels = fig.get_figheight() * fig.dpi
        st_y = t.get_position()[1] - 2.5 * kw['title-font']['fontsize'] / h_pixels
        st = fig.text(.5, st_y, st_prefix + st_score, ha="center", **kw['suptitle-font'])
        fig.canvas.draw()
        bbox = st.get_window_extent(renderer=fig.canvas.get_renderer())
        left, right = bbox.x0 / fig.bbox.width, bbox.x1 / fig.bbox.width
        st.remove()
        fig.text(left, st_y, st_prefix, ha="left", **kw['suptitle-font'])
        fig.text(right, st_y, st_score, c=color, ha="right", **kw['suptitle-font'])
        labels = [k.replace("_", " ").title().replace(" ", "\n") for k in data[0].keys()]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        for d, c1, c2 in zip(data, ["silver", color], ["silver", color]):
            scores = list(d.values())
            scores += scores[:1]
            ax.fill(angles, scores, color=c1, alpha=.6), ax.plot(angles, scores, color=c2, linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, **kw['xlabel-font'])
        ax.set_yticklabels([])
        plt.legend([Line2D([0], [0], c="silver", lw=3), Line2D([0], [0], c=color, lw=3)],
                   ["Non-weighted scores", "Contribution to overall score"],
                   loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, **kw['legend-font'])
        plt.tight_layout(rect=[0, 0, 1, 1. - kw['suptitle-font']['fontsize'] / h_pixels])
        return f"{self._ds.basename}/quality"
    
    @property
    def overall(self):
        scores = {getattr(self, m): w for m, w in self.__w.items() \
                  if self._log.debug(f"computing {m}...") or getattr(self, m) is not None}
        return np.average(list(scores.keys()), weights=list(scores.values()))
    
    @property
    def scores(self):
        return {m: v for m in self.__w.keys() if (v := getattr(self, m)) is not None}
    
    @property
    def weights(self):
        return {m: w for m, w in self.__w.items() if getattr(self, m) is not None}
    
    @weights.setter
    def weights(self, weights):
        if not isinstance(weights, dict):
            self._log.error("weights shall be a dictionary with metric names as keys and weights as values")
            return
        for m in _WEIGHTS.keys():
            if m not in weights:
                self._log.warning(f"weight '{m}' not found in the input dictionary")
        self.__w = weights
    
    # ----------------------------------------------- INDIVIDUAL METRICS -----------------------------------------------
    @cached_property
    def completeness(self):
        """ Score based on the completeness of metadata (absence of missing values). """
        return 1. - (self._ds._data.isnull().mean().mean())
    
    @cached_property
    def consistency(self):
        """Score based on consistency: missing files from data.csv, corrupted files, missing labels. """
        scores, l = [], len(self._ds)
        if self._ds._files:
            scores.append(1. - len([h for h in self._ds._data.hash if not self._ds.files.joinpath(h).is_file()]) / l)
        scores.append(sum(1 if exe.format else 0 for exe in self._ds) / l)  # valid executables
        scores.append(1. - self._ds._data.label.isnull().sum() / l)  # no label
        return np.average(scores)
    
    @cached_property
    def file_balance(self):
        """Score based on architecture, file type, and file size distribution."""
        try:
            return sum(balance(self._ds, f) for f in self.__fbf) / len(self.__fbf)
        except TypeError:
            pass
    
    @cached_property
    def label_balance(self):
        """ Score based on balance between different labels. """
        return balance(self._ds, "label")
    
    @cached_property
    def outliers(self):
        """Score based on files with suspicious size or modified dates. """
        l = len(self._ds)
        suspicious_size  = 1. - (((s := self._ds._data.get("size")) < 1024).sum() + (s > 100 * 1024 * 1024).sum()) / l
        suspicious_mtime = 1. - (pd.to_datetime(self._ds._data.mtime, errors="coerce").dt.year < 2000).sum() / l
        return np.average([suspicious_size, suspicious_mtime])
    
    @cached_property
    def portability(self):
        """ Score based on the presence of .reloc sections and other portability-related PE fields. """
        portabilities = []
        if self._ds._files:
            for exe in self._ds:
                portabilities.append(exe.parsed.portability)
            return sum(portabilities) / len(portabilities) if portabilities else 0.
        else:
            self._log.warning("cannot compute portability as it requires files")
    
    @cached_property
    def similarity(self):
        """ Score based on file similarity using ssdeep. """
        from spamsum import match
        # local iterator for (path, ssdeep), depending on wether ssdeep is part of data.csv
        def _iter():
            try:
                for p, s in zip(self._ds._data.realpath, self._ds._data.ssdeep):
                    yield p, s
            except AttributeError:
                if self._ds._files:
                    for exe in self._ds:
                        yield exe.realpath, exe.ssdeep
                else:
                    self._log.warning("cannot compute similarity as it requires files")
                    return
        # start computing similarity
        ssdeeps, similar_files = {}, 0
        for p, s in _iter():
            for s2, p2 in ssdeeps.items():
                if match(s, s2) >= self.__st:
                    similar_files += 1
                    self._log.debug(f"{s2} ({p2}) similar to {s} ({p})")
            if s not in ssdeeps:
                ssdeeps[s] = p
        return 1. - similar_files / len(self._ds)
    
    @cached_property
    def uniqueness(self):
        """ Score based on how many duplicate files exist (based on the hash used with the Executable abstraction). """
        hashes, duplicates = set(), 0
        if self._ds._files:
            for exe in self._ds:
                if exe.hash in hashes:
                    duplicates += 1
                else:
                    hashes.add(exe.hash)
            return 1 - duplicates / len(self._ds)
        else:
            self._log.warning("cannot compute uniqueness as it requires files")

