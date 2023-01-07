# -*- coding: UTF-8 -*-
import numpy as np
import sklearn
from sklearn.metrics import *
from tinyscript import code, functools

from ..common.config import *


__all__ = ["classification_metrics", "metric_headers", "regression_metrics", "METRIC_DISPLAY", "METRIC_DISPLAYS"]


# metric format function: p=precision, m=multiplier, s=symbol
mformat = lambda p=3, m=1, s=None: lambda x: "-" if x == "-" else ("{:.%df}{}" % p).format(m * x, s or "")
METRIC_CATEGORIES = ["classification", "clustering", "regression"]
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
    'clustering_supervised': {
        'Randomness Score': "nbr",
        'Adjusted Mutual Information Score': "nbr",
        'Homogeneity Completeness V-Measure': "nbr",
    },
    'clustering_unsupervised': {
        'Silhouette Score': "nbr",
        'Calinski Harabasz Score': "nbr",
        'Davies Bouldin Score': "nbr",
    },
    'regression': {
        'MSE': "nbr",
        'MAE': "nbr",
    },
}
METRIC_DISPLAYS = {}
for k in METRIC_CATEGORIES:
    for m, v in METRIC_DISPLAY[k].items():
        METRIC_DISPLAYS[m] = METRIC_DISPLAY.get(v, v)
NO_VALUE = "-"

# add -1 to zero_division possible values
code.replace(sklearn.metrics._classification._check_zero_division,
             "elif isinstance(zero_division, (int, float)) and zero_division in [0, 1]:",
             "elif isinstance(zero_division, (int, float)) and zero_division in [0, 1, -1]:")


def _convert_output(f):
    @functools.wraps(f)
    def _wrapper(*a, **kw):
        pt = kw.get('proctime')
        kw['proctime'] = pt is not None
        r = f(*a, **kw)
        if isinstance(r, (list, tuple)):
            v = list(map(lambda x: NO_VALUE if x == -1 else x, r[0]))
            if pt is not None:
                v.append(pt)
            return (v, ) + [(), r[1:]][len(r) > 1]
        return r
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
    out_arrays, d, r = [], {}, []
    for i, array in enumerate(arrays):
        if array is None:
            out_array = None
        else:
            out_array, d, r = _labels_to_integers(array, d, r)
        out_arrays.append(out_array)
    if len(r) > 0:
        out_arrays = [a if a is None else np.delete(a, r) for a in out_arrays]
    out_arrays.append(d)
    if len([k for k in d.keys() if k != NOT_LABELLED]) <= 2:
        tn, fp, fn, tp = confusion_matrix(*arrays[:2]).ravel()
    else:
        tn, fp, fn, tp = multilabel_confusion_matrix(*arrays[:2], **kwargs).ravel()
    l = kwargs.get('logger')
    if l:
        l.debug("TN: %d ; TP: %d ; FN: %d ; FP: %d" % (tn, tp, fn, fp))
    return out_arrays


def metric_headers(metrics, **kw):
    """ Select the list of headers. """
    try:
        selection = METRIC_DISPLAY[metrics]
    except KeyError:
        raise ValueError("Bad metrics category ; should be one of: %s" % "|".join(METRIC_CATEGORIES))
    for name, func in selection.items():
        selection[name] = METRIC_DISPLAY.get(func, func)
    if kw.get('proctime', False):
        selection['Processing Time'] = METRIC_DISPLAY['ms']
    return selection


@_convert_output
def classification_metrics(y_pred, y_true=None, y_proba=None, labels=None, average="micro", sample_weight=None, **kw):
    """ Compute some classification metrics based on the true and predicted values. """
    # get the true and predicted values without the not-labelled ones and as integers
    yt, yp, ypr, d = _map_values_to_integers(y_true, y_pred, y_proba, **kw)
    if labels is None and d is not None:
        labels = [k for k in d.keys() if k not in [NOT_LABELLED, NOT_PACKED]]
    accuracy = accuracy_score(yt, yp, sample_weight=sample_weight)
    precision, recall, fmeasure, _ = precision_recall_fscore_support(yt, yp, labels=labels or None, average=average,
                                                                     sample_weight=sample_weight)
    mcc = matthews_corrcoef(yt, yp)
    try:
        auc = roc_auc_score(yt, ypr)
    except ValueError:
        auc = -1
    return [accuracy, precision, recall, fmeasure, mcc, auc], metric_headers("classification", **kw)


@_convert_output
def clustering_supervised_metrics(y_pred, y_true=None, **kw):
    """ Compute clustering-related metrics, either based only on predicted values or both true and predicted values. """
    # get the true and predicted values without the not-labelled ones and as integers
    yt, yp, _ = _map_values_to_integers(y_true, y_pred, **kw)
    return [rand_score(yt, yp), adjusted_mutual_info_score(yt, yp), omogeneity_completeness_v_measure(yt, yp)], \
           metric_headers("clustering_supervised", **kw)


@_convert_output
def clustering_unsupervised_metrics(y_pred, y_true=None, **kw):
    """ Compute clustering-related metrics, either based only on predicted values or both true and predicted values. """
    # get the true and predicted values without the not-labelled ones and as integers
    #FIXME: include data (X) in the function's signature
    X = kw['X']
    yt, yp, _ = _map_values_to_integers(y_true, y_pred, **kw)
    return [silhouette_score(X, yp, metric="euclidean"), calinski_harabasz_score(X, yp), davies_bouldin_score(X, yp)], \
           metric_headers("clustering_unsupervised", **kw)


@_convert_output
def regression_metrics(y_pred, y_true=None, **kw):
    """ Compute regression metrics (MSE, MAE) based on the true and predicted values. """
    # get the true and predicted values without the not-labelled ones and as integers
    yt, yp, _ = _map_values_to_integers(y_true, y_pred, **kw)
    return [mean_squared_error(yt, yp), mean_absolute_error(yt, yp)], metric_headers("regression", **kw)

