# -*- coding: UTF-8 -*-
""" Linear preprocessors of scikit-learn (transform of one sample), in their order of operations. """
from math import sqrt


__all__ = ["identity", "maxabs", "minmax", "normalizer", "pca", "robust", "standard"]


def identity(step, x):
    """ FunctionTransformer used by pbox when no preprocessor is selected (pandas.DataFrame -> numpy.array). """
    return list(x)


def minmax(step, x):
    """ MinMaxScaler: X *= scale_ ; X += min_ ; then clip to feature_range if 'clip'. """
    y = [v * s + m for v, s, m in zip(x, step['scale'], step['min'])]
    if step.get('clip'):
        lo, hi = step['feature_range']
        y = [min(max(v, lo), hi) for v in y]
    return y


def standard(step, x):
    """ StandardScaler: X -= mean_ (if with_mean) ; X /= scale_ (if with_std). """
    y = list(x)
    if step.get('mean') is not None:
        y = [v - m for v, m in zip(y, step['mean'])]
    if step.get('scale') is not None:
        y = [v / s for v, s in zip(y, step['scale'])]
    return y


def maxabs(step, x):
    """ MaxAbsScaler: X /= scale_. """
    return [v / s for v, s in zip(x, step['scale'])]


def robust(step, x):
    """ RobustScaler: X -= center_ (if with_centering) ; X /= scale_ (if with_scaling). """
    y = list(x)
    if step.get('center') is not None:
        y = [v - c for v, c in zip(y, step['center'])]
    if step.get('scale') is not None:
        y = [v / s for v, s in zip(y, step['scale'])]
    return y


def normalizer(step, x):
    """ Normalizer (l1, l2 or max norm) ; a null norm leaves the sample unchanged. """
    norm = step['norm']
    if norm == "l1":
        n = sum(abs(v) for v in x)
    elif norm == "l2":
        n = sqrt(sum(v * v for v in x))
    else:
        n = max(abs(v) for v in x)
    n = n or 1.
    return [v / n for v in x]


def pca(step, x):
    """ PCA: (X - mean_) @ components_.T, divided by sqrt(explained_variance_) if whiten. """
    c = [v - m for v, m in zip(x, step['mean'])]
    y = [sum(a * b for a, b in zip(c, comp)) for comp in step['components']]
    if step.get('whiten'):
        y = [v / sqrt(e) for v, e in zip(y, step['explained_variance'])]
    return y
