# -*- coding: UTF-8 -*-
""" Decision trees and random forests (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier). """
import struct


__all__ = ["forest", "tree"]


def _f32(v):
    """ scikit-learn casts the input of trees to float32 before comparing it with the thresholds. """
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _tree_proba(t, x):
    """ Normalized class values of the leaf reached by x (sklearn's predict_proba of a tree, single output). """
    node, left, right, feature, threshold = 0, t['left'], t['right'], t['feature'], t['threshold']
    while left[node] != -1:
        node = left[node] if x[feature[node]] <= threshold[node] else right[node]
    value = t['value'][node]
    normalizer = 0.
    for v in value:
        normalizer += v
    normalizer = normalizer or 1.
    return [v / normalizer for v in value]


def _argmax(p):
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best


def tree(step, x):
    x = [_f32(v) for v in x]
    p = _tree_proba(step['tree'], x)
    return _argmax(p), p


def forest(step, x):
    """ Average of the trees' probabilities, accumulated tree by tree as scikit-learn does. """
    x = [_f32(v) for v in x]
    acc = [0.] * len(step['classes'])
    for t in step['trees']:
        p = _tree_proba(t, x)
        acc = [a + b for a, b in zip(acc, p)]
    n = len(step['trees'])
    p = [a / n for a in acc]
    return _argmax(p), p
