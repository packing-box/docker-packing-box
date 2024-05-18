# -*- coding: UTF-8 -*-
from functools import wraps
from tinyscript.helpers import ints2hex, Path

from ..executable import Executable
from ...helpers import *

lazy_load_module("packer", "pbox.core.items")
lazy_load_module("seaborn")
lazy_load_module("textwrap")


@save_figure
def _characteristic_scatter_plot(dataset, characteristic=None, multiclass=True, **kwargs):
    """ Plot a scatter plot of dataset's reduced data, highlighting the selected characteristic. """
    from ...helpers.figure import plt  # report to get the actual object (cfr lazy loading)
    X, prefix = dataset._data, "bin_" if characteristic == "label" and not multiclass else ""
    if not multiclass:
        X['label'] = X.label.map(LABELS_BACK_CONV).fillna(1).astype('int')
    X_reduced, suffix, meta = reduce_data(X[sorted(dataset._features.keys())], logger=dataset.logger,
                                          return_suffix=True, return_meta=True, **kwargs)
    # define plot
    # important note: 'plt' needs to be called BEFORE 'mpl' ; otherwise, further references to
    #                  'matplotlib' will be seen as 'mpl', causing "ModuleNotFoundError: No module named 'mpl'"
    fig, fsize = plt.figure(figsize=(8, 6)), config['title_font_size'] - 2
    unique_values = np.unique(X[characteristic]).tolist()
    # put not labelled samples above
    try:
        nl = [LABELS_BACK_CONV[NOT_LABELLED], NOT_LABELLED][multiclass]
        unique_values.remove(nl)
        unique_values.append(nl)
    except ValueError:
        pass
    plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = fsize
    plt.suptitle(f"Characteristic '{characteristic}' of dataset {dataset.name}")
    plt.title(", ".join(f"{k}={v}" for k, v in meta.items()))
    # plot a continuous colorbar if the characteristic is continuous and a legend otherwise 
    if len(unique_values) > 6 and characteristic not in ["format", "label", "signature"]:
        sc = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=X[characteristic].to_numpy(), alpha=1.0)
        bbox = plt.get_position()
        width, eps = 0.01, 0.01
        cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
        norm = mpl.colors.Normalize(vmin=X[characteristic].min(), vmax=X[characteristic].max())
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm), cax=cax)
    else:
        for i, value in enumerate(unique_values):
            kw = {'alpha': 1.0, 'label': value}
            if characteristic == "label":
                kw['label'] = READABLE_LABELS(value, binary=not multiclass)
                if multiclass:
                    if value == NOT_LABELLED:
                        kw['c'] = "gray"
                else:
                    kw['c'] = {-1: "gray", 0: "green", 1: "red"}[value]
            plt.scatter(X_reduced[X[characteristic] == value, 0], X_reduced[X[characteristic] == value, 1], **kw)
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fsize)  
    return f"{dataset.basename}/characteristic/{prefix}{characteristic}{suffix}"


@save_figure
def _features_bar_chart(dataset, feature=None, num_values=None, multiclass=False, scaler=None, **kw):
    """ Plot the distribution of the given feature or multiple features combined. """
    l = dataset.logger
    if feature is None:
        l.warning("No feature provided, stopping.")
        return  # no feature to handle
    from ...helpers.figure import plt  # report to get the actual object (cfr lazy loading)
    from sklearn.covariance import empirical_covariance
    from sklearn.preprocessing import MinMaxScaler
    scaler = scaler or MinMaxScaler
    # data preparation
    feature = filter_features(dataset, feature)
    l.info(f"Counting values for feature{['', 's'][len(feature) > 1]} {', '.join(feature)}...")
    #FIXME: for continuous values, convert to ranges to limit chart's height
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
        var = "Variances:\n- " + "\n- ".join(f"{f}: {cov_matrix[i][i]:.03f}" for i, f in enumerate(feature))
        covar = "Covariances:\n"
        for i in range(len(cov_matrix)):
            for j in range(i + 1, len(cov_matrix)):
                covar += f"- {feature[i]} / {feature[j]}: {cov_matrix[i][j]:.03f}\n"
    else:
        var = f"Variance: {cov_matrix[0][0]:.03f}"
    # be sure to have values for every label (it was indeed not seen if 0, so set the default value)
    for v, d in counts.items():
        if multiclass:
            for lbl in labels:
                d.setdefault(lbl, 0)
        else:
            d.setdefault('Packed', 0)
    # merge counts of not packed and other counts
    all_counts = {k: {'Not packed': v} for k, v in sorted(counts_np.items(), key=lambda x: x[0])}
    for k, v in counts.items():
        for sk, sv in v.items():
            all_counts[k][sk] = sv  # force keys order
    counts = all_counts
    if num_values:
        l.debug(f"selecting {num_values} most occurring feature values...")
        counts = dict(sorted(counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:num_values])
    l.debug("sorting feature values...")
    # sort counts by feature value and by label
    counts = {k: {sk: sv for sk, sv in sorted(v.items(), key=lambda x: x[0].lower())} \
              for k, v in sorted(counts.items(), key=lambda x: x[0])}
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
                "\n".join(textwrap.wrap(f"combination of {', '.join(dataset._features[f] for f in feature)}", 60))
        title = title[0].upper() + title[1:] + f" for dataset {dataset.name}"
    except KeyError as e:
        l.error(f"Feature '{e.args[0]}' does not exist in the target dataset.")
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
    x_label, y_label = f"Samples [%] for the selected feature{plur}", \
                       f"Feature{plur} values" + (f" (top {num_values})" if num_values else "")
    yticks = [str(k[0]) if isinstance(k, (tuple, list)) and len(k) == 1 else str(k) \
              for k in counts.keys()]
    plt.rcParams['font.family'] = ["serif"]
    plt.figure(figsize=(6, (len(title.splitlines()) * 24 + 11 * len(counts) + 120) / 80))
    plt.title(title, pad=20, fontweight="bold", fontsize=16)
    plt.xlabel(x_label, fontdict={'size': 14})
    plt.ylabel(y_label, fontdict={'size': 14})
    starts = [0 for i in range(len(values[0]))]
    for p, lb ,c, v in zip(percentages, labels, cmap, values):
        b = plt.barh(yticks, p, label=lb, color=c, left=starts)
        starts = [x + y for x, y in zip(starts, p)]
        plt.bar_label(b, labels=["" if x == 0 else x for x in v], label_type="center", color="white")
    plt.yticks(**({'family': "serif", 'fontsize': 14} if vtype is hex else {'fontsize': 14}))
    plt.legend()
    return f"{dataset.basename}/features/{['', 'combo-'][len(feature) > 1]}{feature[0]}"


@save_figure
def _features_comparison_heatmap(dataset, datasets=None, feature=None, max_features=None,
                                 aggregate="byte_[0-9]+_after_ep", **kw):
    """ Plot a heatmap with the diffferences of feature values between a reference dataset (Dataset instance) and the
         given list of datasets (by name). """
    from ...helpers.figure import plt  # report to get the actual object (cfr lazy loading)
    from sklearn.preprocessing import StandardScaler
    l = dataset.logger
    datasets_feats = {dataset.basename: dataset._data.copy()}
    for ds in (datasets or []):
        datasets_feats[ds.basename] = ds._data.copy()
    feature = filter_features(dataset, feature or "*")
    df = pd.concat(datasets_feats.values(), keys=datasets_feats.keys(), names=['experiment', 'hash'])[feature] \
         .astype('float')
    scaler_v = StandardScaler().fit(df.loc[dataset.basename][feature])
    df_rank = pd.DataFrame(scaler_v.transform(df[feature]), columns=df.columns, index=df.index)
    df = df - df.loc[dataset.basename]
    df_rank = df_rank - df_rank.loc[dataset.basename]
    pivoted_rank = df_rank.dropna().pivot_table(index='experiment', values=feature, aggfunc=np.mean)[feature]
    pivoted = df.dropna().pivot_table(index='experiment', values=feature, aggfunc=np.mean)[feature]
    pivoted_rank = pivoted_rank.drop(index=dataset.basename)
    pivoted = pivoted.drop(index=dataset.basename)
    if max_features in [None, 0] or max_features > len(feature):
        max_features = len(feature)
    if aggregate is not None:
        to_group = list(pivoted.columns.str.extract("(" + aggregate + ")",expand=False).dropna())
        pivoted[aggregate + "_mean"] = pivoted[to_group].mean(axis=1)
        pivoted_rank[aggregate + "_mean"] = pivoted_rank[to_group].mean(axis=1)
        pivoted_rank = pivoted_rank.drop(to_group, axis=1)
        pivoted = pivoted.drop(to_group, axis=1)
        if max_features > pivoted.shape[1]:
            max_features = pivoted.shape[1]
    order = sorted(sorted(pivoted_rank.columns, key=lambda x: abs(pivoted_rank[x]).mean())[-max_features:],
                   key=lambda x: pivoted_rank[x].mean())[::-1]
    ticks = [-3, 0 , 3]
    label = f"Feature value difference from {dataset.basename}"
    title = f"Feature value comparison with {dataset.basename}"
    plt.figure(figsize=(1.2*len(pivoted_rank.index) + 3 , round(0.25*max_features + 2.2)), dpi=200)
    annotations = pivoted[order].applymap(lambda x: f"{x:.2f}" if abs(x) < 1000 else f"{x:.1e}").values.T
    ax = seaborn.heatmap(data=pivoted_rank[order].values.T, annot=annotations, fmt='s', cmap='vlag', linewidth=.5,
                         xticklabels=pivoted_rank.index, yticklabels=order, vmin=ticks[0], vmax=ticks[2],
                         cbar_kws = {'location':'right', 'ticks': ticks, 'label':label}, linecolor='black')
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.title(title)
    ax.collections[0].colorbar.set_ticklabels(["Negative", "Negligible", "Positive"])
    plt.tight_layout()
    return f"{dataset.basename}/features-compare/{'-'.join(datasets_feats)}"


@save_figure
def _information_gain_bar_chart(dataset, feature=None, max_features=None, multiclass=False, **kw):
    """ Plot a bar chart of the information gain of features in descending order. """
    from ...helpers.figure import plt  # report to get the actual object (cfr lazy loading)
    from sklearn.feature_selection import mutual_info_classif as mic
    l = dataset.logger
    feature = filter_features(dataset, feature)
    feats = dataset._data.copy()
    feats = feats.set_index("hash")
    feats = feats[feats['label'] != NOT_LABELLED] # skip not-labelled samples
    labels = feats['label'] if multiclass else feats['label'] == NOT_PACKED
    feats = feats[feature]
    info = mic(feats, labels, n_neighbors=5)
    if max_features in [None, 0] or max_features > len(feature):
        max_features = len(feature)
    l.debug("plotting figure...")
    # feature ranking
    indices = range(len(info))
    order = sorted(
        sorted(indices, key=lambda x: abs(info[x]))[-max_features:],
        key=lambda x: info[x])
    # plot
    plt.figure(figsize=(10, round(0.25*max_features + 2.2)), dpi=200)
    plt.barh(*zip(*[(feature[x], info[x]) for x in order]), height=.5)
    plt.title(f"Mutual information for dataset {dataset.name}")
    plt.ylabel('Features')
    plt.xlabel('Mutual information')
    plt.yticks(rotation='horizontal')
    plt.margins(y=1/max_features)
    plt.axvline(x=0, color='k')
    return f"{dataset.basename}/infogain"


@save_figure
def _information_gain_comparison_heatmap(dataset, datasets=None, feature=None, max_features=None, multiclass=False,
                                         aggregate="byte_[0-9]+_after_ep", **kw):
    """ Plot a heatmap with the diffferences of information gain between a reference dataset (Dataset instance) and the
         given list of datasets (by name). """
    #FIXME: remove temp coode [START]
    import matplotlib
    #Create the style of the font 
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 10}         
    matplotlib.rc('font', **font) #set the font style created
    #FIXME: remove temp coode [END]
    from ...helpers.figure import plt  # report to get the actual object (cfr lazy loading)
    from sklearn.feature_selection import mutual_info_classif as mic
    from sklearn.impute import SimpleImputer
    l = dataset.logger
    datasets_feats = {dataset.basename: dataset._data.copy()}
    for ds in (datasets or []):
        datasets_feats[ds.name] = ds._data.copy()
    feature = filter_features(dataset, feature or "*")
    df = pd.concat(datasets_feats.values(), keys=datasets_feats.keys(), names=['experiment', 'hash'])
    df = df[df['label'] != NOT_LABELLED]
    df = filter_data(df, feature=feature)
    #fct = [lambda x: pd.Series(mic(x[feature].astype('float'), x['label'] == NOT_PACKED), index=feature),
    #       lambda x: pd.Series(mic(x[feature].astype('float'), x['label']), index=feature)][multiclass]
    df[feature] = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(df[feature])
    df = df.groupby('experiment').apply(lambda x: pd.Series(mic(x[feature].astype('float'), x['label'], random_state=42), index=feature))
    df = (df - df.loc[dataset.basename]).apply(abs)
    df = df.drop(index=dataset.basename)
    if max_features in [None, 0] or max_features > len(feature):
        max_features = len(feature)
    if aggregate is not None:
        to_group = list(df.columns.str.extract("(" + aggregate + ")", expand=False).dropna())
        df[aggregate + "_mean"] = df[to_group].mean(axis=1)
        df = df.drop(to_group, axis=1)
        if max_features > df.shape[1]:
            max_features = df.shape[1]
    order = sorted(sorted(df.columns, key=lambda x: df[x].mean())[-max_features:],
                   key=lambda x: df[x].mean())[::-1]
    # normalize per column
    df[order] = df[order].div(df[order].max(axis=0), axis=1)
    label = f"Normalized IG difference from reference dataset"
    title = f"Information Gain comparison with reference dataset {dataset.basename}"
    plt.figure(figsize=(1.2*len(df.index) + 3, round(0.25 * max_features + 2.2)), dpi=200)
    annotations = df[order].applymap(lambda x: "%.2f"%x if abs(x) < 1000 else "%.1e"%x).values.T
    ax = seaborn.heatmap(data=df[order].values.T, annot=annotations, fmt="s", xticklabels=df.index, yticklabels=order,
                         cmap="YlOrBr", linewidth=0, cbar_kws={'location': 'right', 'ticks': [0,1], 'label': label},
                         vmin=0, vmax=1)
    ax.xaxis.tick_top()
    plt.title(title, fontfamily="serif", fontsize=16, fontweight="bold", pad=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    dataset._temp_df = df
    return f"{dataset.basename}/infogain-compare/{'-'.join(datasets_feats)}"


@save_figure
def _labels_pie_chart(dataset, **kw):
    """ Describe the dataset with a pie chart. """
    from ...helpers.figure import plt  # report to get the actual object (cfr lazy loading)
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
    perc = {k: f"{100*v/tot:.1f}%" for k, v in c.items()}
    labels = [packer.Packer.get(k).cname.replace("_", " ") if i >= n else \
              {NOT_LABELLED: "Not labelled", NOT_PACKED: "Not packed"}[k] for i, k in enumerate(classes)]
    # plot
    l.debug("plotting figure...")
    plt.figure(figsize=(8, 4))
    plt.title(f"Distribution of labels for dataset {dataset.name}", pad=10, fontweight="bold")
    # - draw a first pie with white labels on the wedges
    plt.pie([c[k] for k in classes], colors=cmap, startangle=180, radius=.8,
            autopct=lambda p: "{:.1f}%\n({:.0f})".format(p, p/100*tot),
            textprops={'color': "white", 'fontsize': "small"})
    # - draw a second pie, transparent, just to use black labels
    for wedge in plt.pie([c[k] for k in classes], labels=labels, labeldistance=1, startangle=180)[0]:
        wedge.set_alpha(0.)
    return f"{dataset.basename}/labels"


@save_figure
def _samples_individual_visualization(dataset, query=None, n=0, **kw):
    if not dataset._files:
        dataset.logger.warning("Plotting individual samples only works for datasets with files")
        return
    for e in filter_data_iter(dataset._data, query, limit=n or 0, logger=dataset.logger):
        exe = Executable(dataset=dataset, hash=e.hash)
        dataset.logger.info(f"Plotting {exe.basename} ({Path(exe.realpath).basename})...")
        yield exe.plot(**kw)


_PLOTS = {
    'characteristic':   _characteristic_scatter_plot,
    'features':         _features_bar_chart,
    'features-compare': _features_comparison_heatmap,
    'infogain':         _information_gain_bar_chart,
    'infogain-compare': _information_gain_comparison_heatmap,
    'labels':           _labels_pie_chart,
    'samples':          _samples_individual_visualization,
}

