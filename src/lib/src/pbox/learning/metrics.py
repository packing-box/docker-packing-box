# -*- coding: UTF-8 -*-
import numpy as np
import sklearn
from sklearn.metrics import *
from tinyscript import code, functools, re

from ..common.config import *


__all__ = ["classification_metrics", "clustering_metrics", "regression_metrics",
           "highlight_best", "metric_headers", "METRIC_DISPLAY"]


# metric format function: p=precision, m=multiplier, s=symbol
mformat = lambda p=3, m=1, s=None: lambda x: "-" if x == "-" else ("{:.%df}{}" % p).format(m * x, s or "")
METRIC_DISPLAY = {
    # helpers
    '%':   mformat(2, 100, "%"),
    'ms':  mformat(3, 1000, "ms"), # used for 'Processing Time' in metric_headers(...)
    'nbr': mformat(),
    'classification': {
        'Accuracy':  "%",
        'Precision': "%",
        'Recall':    "%",
        'F-Measure': "%",
        'MCC':       "%",
        'AUC':       "%",
    },
    'clustering': {
        # labels known
        'Randomness Score': "nbr",
        'Adjusted Mutual Information Score': "nbr",
        'Homogeneity Completeness V-Measure': "nbr",
        # labels not known
        'Silhouette Score': "nbr",
        'Calinski Harabasz Score': "nbr",
        'Davies Bouldin Score': "nbr",
    },
    'regression': {
        'MSE': "nbr",
        'MAE': "nbr",
    },
}
NO_VALUE = "-"

_METRIC_CATEGORIES = ["classification", "clustering", "regression"]
_METRIC_DISPLAYS = {}
for k in _METRIC_CATEGORIES:
    for m, v in METRIC_DISPLAY[k].items():
        _METRIC_DISPLAYS[m] = METRIC_DISPLAY.get(v, v)

# add -1 to zero_division possible values
code.replace(sklearn.metrics._classification._check_zero_division,
             "elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:",
             "elif isinstance(zero_division, (int, float)) and zero_division in [0, 1, -1]:")


def _convert_output(f):
    @functools.wraps(f)
    def _wrapper(X, yp, *a, **kw):
        l, pt = kw.get('logger', null_logger), kw.get('proctime')
        kw['proctime'] = pt is not None
        if all(x == LABELS_BACK_CONV[NOT_LABELLED] for x in yp):
            l.warning("No prediction (returned values mean 'not labelled'), hence no %s metric" % \
                      f.__name__.split("_")[0])
            return
        r = f(*a, **kw)
        if isinstance(r, (list, tuple)):
            v = list(map(lambda x: NO_VALUE if x == -1 else x, r[0]))
            if pt is not None:
                v.append(pt)
            return (v, ) + [(), r[1:]][len(r) > 1]
        return r
    return _wrapper


def _skip_if_labels_ignored(f):
    @functools.wraps(f)
    def _wrapper(*a, **kw):
        l, ignore = kw.get('logger', null_logger), kw.get('ignore_labels')
        if ignore:
            l.debug("> labels ignored, skipping %s metrics..." % f.__name__.split("_")[0])
            return
        return f(*a, **kw)
    return _wrapper


def _labels_to_integers(array, transdict=None, not_labelled_idx=None):
    labels = np.unique(array).tolist()
    def __rm(v):
        try:
            labels.remove(v)
        except ValueError:
            pass
    # if the array contains floats (i.e. a vector of prediction probabilities), do nothing
    if any(isinstance(x, float) for x in labels):
        return array, transdict, not_labelled_idx
    # if the array only contains integers already, update the list of indices for not labelled samples
    if all(isinstance(x, int) for x in labels):
        not_labelled_idx = list(set((not_labelled_idx or []) + list(np.where(array == -1)[0])))
        return array, transdict, not_labelled_idx
    # if no translation dictionary was provided, compute it based on the collected labels
    if transdict is None:
        transdict = {NOT_LABELLED: -1, NOT_PACKED: 0}
        for v in transdict.keys():
            __rm(v)
        for i, v in enumerate(labels, 1):
            transdict[v] = i
    # now complete eventually missing labels
    for v in transdict.keys():
        __rm(v)
    if len(labels) > 0:
        for i, v in enumerate(labels, len(labels)):
            transdict[v] = i
    # map the values from the translation dictionary to the input array
    sorted_idx = np.argsort(transdict.keys())
    idx = np.searchsorted(transdict.keys(), array, sorter=sorted_idx)
    new_array = np.asarray(transdict.values())[sorted_idx][idx]
    return new_array, transdict, list(set((not_labelled_idx or []) + list(np.where(new_array == -1)[0])))


def _map_values_to_integers(*arrays, **kwargs):
    """ Map values from the given arrays to integers. """
    l = kwargs.get('logger', null_logger)
    out_arrays, d, r = [], {}, []
    l.debug("> filtering out not labelled samples...")
    for i, array in enumerate(arrays):
        if array is None:
            out_array = None
        else:
            out_array, d, r = _labels_to_integers(array, d, r)
        out_arrays.append(out_array)
    if len(r) > 0:
        out_arrays = [a if a is None else np.delete(a, r) for a in out_arrays]
    out_arrays.append(d)
    binary = len([k for k in d.keys() if k != NOT_LABELLED]) <= 2
    kw = {k: v for k, v in kwargs.items() if k in ["sample_weight", "labels", "samplewise"]}
    for i, matrix in enumerate(multilabel_confusion_matrix(*arrays[:2], **kw)):
        if binary and i == 0:
            continue
        tn, fp, fn, tp = matrix.ravel()
        l.debug("> %sTN: %d ; TP: %d ; FN: %d ; FP: %d" % (["[%d] " % i, ""][binary], tn, tp, fn, fp))
    return out_arrays


def highlight_best(data, headers=None, exclude_cols=[0, -1], formats=_METRIC_DISPLAYS):
    """ Highlight the highest values in the given table. """
    if len(data[0]) != len(headers):
        raise ValueError("headers and row lengths mismatch")
    ndata, exc_cols = [], [x % len(headers) for x in exclude_cols]
    maxs = [None if i in exc_cols else 0 for i, _ in enumerate(headers)]
    fl = lambda f: -1. if f == "-" else float(f)
    # search for best values
    for d in data:
        for i, v in enumerate(d):
            if maxs[i] is None:
                continue
            maxs[i] = max(maxs[i], fl(v))
    # reformat the table, setting bold text for best values
    for d in data:
        ndata.append([bold((formats or {}).get(k, lambda x: x)(v)) if maxs[i] and fl(v) == maxs[i] else \
                     (formats or {}).get(k, lambda x: x)(v) for i, (k, v) in enumerate(zip(headers, d))])
    return ndata


def metric_headers(metrics, **kw):
    """ Select the list of headers, including the processing time if specified. """
    try:
        selection = METRIC_DISPLAY[metrics]
    except KeyError:
        raise ValueError("Bad metrics category ; should be one of: %s" % "|".join(_METRIC_CATEGORIES))
    for name, func in selection.items():
        selection[name] = METRIC_DISPLAY.get(func, func)
    patterns = kw.get('include')
    if patterns:
        selection = {n: m for n, m in selection.items() if any(re.match(p, n) for p in patterns)}
    if kw.get('proctime', False):
        selection['Processing Time'] = METRIC_DISPLAY['ms']
    return selection


@_skip_if_labels_ignored
@_convert_output
def classification_metrics(X, y_pred, y_true=None, y_proba=None, labels=None, sample_weight=None, ignore_labels=False,
                           **kw):
    """ Compute some classification metrics based on the true and predicted values. """
    l = kw.get('logger', null_logger)
    if ignore_labels:
        l.debug("> labels ignored, skipping classification metrics...")
        return
    binary = len(set([k for k in y_pred if k != NOT_LABELLED])) <= 2 and \
             len(set([k for k in y_true if k != NOT_LABELLED])) <= 2
    # get the true and predicted values without the not-labelled ones and as integers
    yt, yp, ypr, d = _map_values_to_integers(y_true, y_pred, y_proba, **kw)
    if labels is None and d is not None:
        labels = [k for k in d.keys() if k not in [NOT_LABELLED, NOT_PACKED]]
    accuracy = accuracy_score(yt, yp, sample_weight=sample_weight)
    precision, recall, fmeasure, _ = precision_recall_fscore_support(yt, yp, labels=labels or None,
                                                                     average=["weighted", None][binary],
                                                                     sample_weight=sample_weight)
    if binary:
        precision, recall, fmeasure = precision[1], recall[1], fmeasure[1]
    mcc = matthews_corrcoef(yt, yp)
    try:
        auc = roc_auc_score(yt, ypr)
    except (TypeError, ValueError):
        auc = -1
    return [accuracy, precision, recall, fmeasure, mcc, auc], metric_headers("classification", **kw)


@_convert_output
def clustering_metrics(X, y_pred, y_true=None, ignore_labels=False, **kw):
    """ Compute clustering-related metrics based on the input data and the true and predicted values. """
    l = kw.get('logger', null_logger)
    # labels not known: no mapping to integers and filtering of not-labelled values as we only consider predicted ones
    if ignore_labels or y_true is None or all(y == NOT_LABELLED for y in y_true):
        if ignore_labels:
            l.debug("> labels ignored, skipping label-dependent clustering metrics...")
        return [silhouette_score(X, y_pred, metric="euclidean"), calinski_harabasz_score(X, y_pred), \
                davies_bouldin_score(X, y_pred)], metric_headers("clustering",
                                                                 include=["Silhouette", "Calinski", "Davies"], **kw)
    # labels known: get the true and predicted values without the not-labelled ones and as integers
    yt, yp, _ = _map_values_to_integers(y_true, y_pred, **kw)
    print(y_true, y_pred)
    return [rand_score(yt, yp), adjusted_mutual_info_score(yt, yp), homogeneity_completeness_v_measure(yt, yp), \
            silhouette_score(X, y_pred, metric="euclidean"), calinski_harabasz_score(X, y_pred), \
            davies_bouldin_score(X, y_pred)], metric_headers("clustering", **kw)


@_skip_if_labels_ignored
@_convert_output
def regression_metrics(X, y_pred, y_true=None, ignore_labels=False, **kw):
    """ Compute regression metrics (MSE, MAE) based on the true and predicted values. """
    # get the true and predicted values without the not-labelled ones and as integers
    yt, yp, _ = _map_values_to_integers(y_true, y_pred, **kw)
    return [mean_squared_error(yt, yp), mean_absolute_error(yt, yp)], metric_headers("regression", **kw)

