# -*- coding: UTF-8 -*-
""" Multi-Layer Perceptron classifier (MLPClassifier): forward pass and decision as scikit-learn computes them. """
from math import exp, tanh


__all__ = ["mlp"]


def _logistic(v):
    return 1. / (1. + exp(-v)) if v >= -700 else 0.


_ACTIVATIONS = {
    'identity': lambda v: v,
    'logistic': _logistic,
    'relu':     lambda v: v if v > 0 else 0.,
    'tanh':     tanh,
}


def mlp(step, x):
    a = list(x)
    layers = list(zip(step['coefs'], step['intercepts']))
    for n, (w, b) in enumerate(layers):
        # a @ W + b, with W of shape (n_in, n_out)
        z = [sum(a[i] * w[i][j] for i in range(len(a))) + b[j] for j in range(len(b))]
        if n < len(layers) - 1:
            a = [_ACTIVATIONS[step['activation']](v) for v in z]
        else:
            a = z
    out = step['out_activation']
    if out == "logistic":   # binary: one output, threshold at 0.5 (LabelBinarizer.inverse_transform)
        p = _logistic(a[0])
        return int(p > .5), [1. - p, p]
    if out == "softmax":
        m = max(a)
        e = [exp(v - m) for v in a]
        s = sum(e)
        p = [v / s for v in e]
        best = max(range(len(p)), key=lambda i: (p[i], -i))
        return best, p
    raise ValueError(f"unsupported output activation '{out}'")
