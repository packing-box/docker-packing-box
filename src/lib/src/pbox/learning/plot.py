# -*- coding: UTF-8 -*-
from tinyscript import code, colored
from tinyscript.helpers import ints2hex, lazy_load_module, Path

from ..common.config import *
from ..common.utils import *

lazy_load_module("packer", "pbox.items")
lazy_load_module("pandas", alias="pd")
lazy_load_module("textwrap")


__all__ = ["plot", "PLOTS"]


@figure_path
def _dataset_information_gain_bar_chart(dataset, feature=None, format="png", filter_zero_ig=True, **kw):
    """ Plot a bar chart of the information gain of features in descending order. """
    l = dataset.logger
    feature = select_features(dataset, feature)
    # plot
    l.debug("plotting figure...")
    plt.figure(figsize=(8, 0))
    #TODO
    return "%s_infogain.%s" % (dataset.basename, format)


@figure_path
def _dataset_labels_pie_chart(dataset, format="png", **kw):
    """ Describe the dataset with a pie chart. """
    l = dataset.logger
    # data preparation
    l.debug("collecting label counts...")
    c = {k: v for k, v in dataset._metadata['counts'].items()}
    c.setdefault(NOT_LABELLED, 0)
    c.setdefault(NOT_PACKED, 0)
    classes, cmap, n = [], [], 0
    if c[NOT_LABELLED] > 0:
        classes.append(NOT_LABELLED)
        cmap.append("gray")
        n += 1
    if c[NOT_PACKED] > 0:
        classes.append(NOT_PACKED)
        cmap.append("green")
        n += 1
    classes += [k for k in dataset._metadata['counts'] if k not in [NOT_LABELLED, NOT_PACKED]]
    cmap += [list(COLORMAP.keys())[i % len(COLORMAP)] for i in range(len(classes) - n)]
    tot = sum(c.values())
    perc = {k: "%.1f%%" % (100 * v / tot) for k, v in c.items()}
    labels = [packer.Packer.get(k).cname.replace("_", " ") if i >= n else \
              {NOT_LABELLED: "Not labelled", NOT_PACKED: "Not packed"}[k] for i, k in enumerate(classes)]
    # plot
    l.debug("plotting figure...")
    plt.figure(figsize=(8, 4))
    plt.title("Distribution of labels for dataset %s" % dataset.name, pad=10, fontweight="bold")
    # - draw a first pie with white labels on the wedges
    plt.pie([c[k] for k in classes], colors=cmap, startangle=180, radius=.8,
            autopct=lambda p: "{:.1f}%\n({:.0f})".format(p, p/100*tot), textprops={'color': "white", 'fontsize': 8})
    # - draw a second pie, transparent, just to use black labels
    for wedge in plt.pie([c[k] for k in classes], labels=labels, labeldistance=1, startangle=180)[0]:
        wedge.set_alpha(0.)
    return "%s_labels.%s" % (dataset.basename, format)


@figure_path
def _dataset_features_bar_chart(dataset, feature=None, multiclass=False, format="png", scaler=None, **kw):
    """ Plot the distribution of the given feature or multiple features combined. """
    if feature is None: 
        return  # no feature to handle
    from sklearn.covariance import empirical_covariance
    from sklearn.preprocessing import MinMaxScaler
    scaler = scaler or MinMaxScaler
    l = dataset.logger
    # data preparation
    feature = select_features(dataset, feature)
    l.info("Counting values for feature%s %s..." % (["", "s"][len(feature) > 1], ", ".join(feature)))
    # start counting, keeping 'Not packed' counts separate (to prevent it from being sorted with others)
    counts_np, counts, labels, data = {}, {}, [], pd.DataFrame()
    for exe in dataset:
        row = {f: v for f, v in exe.data.items() if f in feature}
        data = pd.concat([data, pd.DataFrame.from_records([row])], ignore_index=True)
        v = tuple(row.values())
        counts_np.setdefault(v, 0)
        counts.setdefault(v, {} if multiclass else {'Packed': 0})
        lbl = str(exe.label)
        if lbl == NOT_PACKED:
            counts_np[v] += 1
        elif multiclass:
            lbl = packer.Packer.get(lbl).cname
            counts[v].setdefault(lbl, 0)
            counts[v][lbl] += 1
            if lbl not in labels:
                labels.append(lbl)
        else:
            counts[v]['Packed'] += 1
    data = scaler().fit_transform(data)
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
    try:
        title = dataset._features[feature[0]] if len(feature) == 1 else \
                "\n".join(textwrap.wrap("combination of %s" % ", ".join(dataset._features[f] for f in feature), 60))
        title = title[0].upper() + title[1:] + " for dataset %s" % dataset.name
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
    font = {'color': "lightgray", 'size': 10}
    yticks = [str(k[0]) if isinstance(k, (tuple, list)) and len(k) == 1 else str(k) \
              for k in counts.keys()]
    plt.figure(figsize=(8, (len(title.splitlines()) * 24 + 11 * len(counts) + 120) / 80))
    plt.title(title, pad=20, fontweight="bold")
    plt.xlabel(x_label, fontdict=font)
    plt.ylabel(y_label, fontdict=font)
    starts = [0 for i in range(len(values[0]))]
    for p, lb ,c, v in zip(percentages, labels, cmap, values):
        b = plt.barh(yticks, p, label=lb, color=c, left=starts)
        starts = [x + y for x, y in zip(starts, p)]
        plt.bar_label(b, labels=["" if x == 0 else x for x in v], label_type="center", color="white")
    plt.yticks(**({'family': "monospace", 'fontsize': 8} if vtype is hex else {'fontsize': 9}))
    plt.legend()
    return dataset.basename + "_features_" + ["", "combo-"][len(feature) > 1] + feature[0] + "." + format


def plot(obj, ptype, dpi=200, **kw):
    """ Generic plot function. """
    try:
        with Path("~/.packing-box/experiment.env", expand=True).open() as f:
            root = Path(f.read().strip()).joinpath("figures")
        root.mkdir(exist_ok=True)
    except FileNotFoundError:
        root = Path(".")
    obj.logger.info("Preparing data...")
    try:
        img = root.joinpath(PLOTS[ptype](obj, **kw))
    except KeyError:
        obj.logger.error("Plot type does not exist (should be one of [%s])" % "|".join(PLOTS.keys()))
        return
    except TypeError:
        return
    obj.logger.info("Saving to %s..." % img)
    import matplotlib.pyplot
    matplotlib.pyplot.savefig(img, format=kw.get('format'), dpi=dpi, bbox_inches="tight")


PLOTS = {
    'ds-infogain': _dataset_information_gain_bar_chart,
    'ds-labels':   _dataset_labels_pie_chart,
    'ds-features': _dataset_features_bar_chart,
}
