# -*- coding: UTF-8 -*-
""" Inference of trained models in pure Python, from their parameters as plain data (see pbox.core.model.export).

A model is a dict: {'steps': [transformer steps..., classifier step]} where each step is a dict with a 'type' key. The
 functions of the submodules reproduce scikit-learn's computations (operation order, float32 casts of the trees), so
 that predictions are identical and probabilities equal up to floating-point summation order where BLAS is involved.
 Only the submodules a model needs are embedded in an exported tool, hence the imports in the dispatching functions.
"""

__all__ = ["predict"]


def transform(step, x):
    t = step['type']
    if t in ("identity", "minmax", "standard", "maxabs", "robust", "normalizer", "pca"):
        from . import preprocessing
        return getattr(preprocessing, t)(step, x)
    raise ValueError(f"unsupported transformer '{t}'")


def classify(step, x):
    """ (index of the predicted class, class probabilities or None). """
    t = step['type']
    if t in ("tree", "forest"):
        from . import tree
        return getattr(tree, t)(step, x)
    if t == "svc":
        from . import svm
        return svm.svc(step, x)
    if t == "mlp":
        from . import mlp
        return mlp.mlp(step, x)
    raise ValueError(f"unsupported classifier '{t}'")


def predict(model, x):
    """ Predict the class of one sample (list of floats, in the order of model['features']).

    Returns (class, probabilities as {class: probability} or None if the model does not give any). """
    for step in model['steps'][:-1]:
        x = transform(step, x)
    clf = model['steps'][-1]
    idx, proba = classify(clf, x)
    classes = clf['classes']
    return classes[idx], None if proba is None else dict(zip(classes, proba))
