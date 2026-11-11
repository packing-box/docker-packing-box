# -*- coding: UTF-8 -*-
""" Evaluation of the feature expressions of the YAML definitions, as pbox evaluates them (see
     pbox.helpers.items.dict2.__call__ and pbox.core.executable.features.Features).

Features are given as definitions {name: {'expr': ..., 'boolean': ..., 'fail': ...}} with the expression to evaluate for
 the format of the executable (the "*(" failsafe markers already rewritten). They are computed in pbox's order:
  1. the features that only need raw extracted data (e.g. pefeats' values) ; an error then aborts the extraction ;
  2. the others, until convergence, with the parsed binary and the constants ; an error then follows the 'fail' setting
      of the feature: "error" aborts the extraction, "warning" leaves the feature undefined, "continue" sets it to None.
 In both cases, a division by zero sets the feature to None, and boolean features (named "is_*" or "has_*") are
 converted with bool().
"""
import inspect
import statistics
import string


__all__ = ["compute", "namespace", "ExtractionError", "HELPERS"]


class ExtractionError(ValueError):
    """ The features of the executable cannot be computed (as pbox would fail to compute them). """


def _fail_safe(f):
    def _wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            return
    return _wrapper


def _apply(f):
    def _wrapper(op):
        def _subwrapper(*a, **kw):
            return op(f(*a, **kw))
        return _subwrapper
    return _wrapper


def _ent(s, *a, **kw):
    if a or kw:
        raise ValueError("ent() with parameters is not supported")
    from .primitives import entropy
    return entropy(s.encode() if isinstance(s, str) else s)


# helpers of pbox's evaluation namespace (pbox.helpers.items) that can be used without pbox
HELPERS = {
    'apply':     _apply,
    'ent':       _ent,
    'failsafe':  _fail_safe,
    'find':      str.find,
    'hex2bytes': bytearray.fromhex,
    'isin':      lambda s, l: s in l,
    'joinstr':   "".join,
    'last':      lambda o, sep=".": str(o).split(sep)[-1],
    'len':       lambda i: sum(1 for _ in i) if inspect.isgenerator(i) else len(i),
    'lower':     lambda s: s.lower(),
    'max':       lambda l, *a, **kw: None if len(l2 := [x for x in l if x is not None]) == 0 else max(l2, *a, **kw),
    'min':       lambda l, *a, **kw: None if len(l2 := [x for x in l if x is not None]) == 0 else min(l2, *a, **kw),
    'printable': string.printable,
    'repeatn':   lambda s, n: (s * (n // len(s) + 1))[:n],
    'value':     lambda o: getattr(o, "value", o),
}
for _k in ["abs", "all", "any", "bool", "divmod", "float", "hash", "hex", "id", "int", "list", "map", "next", "oct",
           "ord", "pow", "range", "round", "set", "str", "sum", "tuple", "type"]:
    HELPERS[_k] = __builtins__[_k] if isinstance(__builtins__, dict) else getattr(__builtins__, _k)
for _k in dir(statistics):
    if not _k.startswith("_") and inspect.isfunction(getattr(statistics, _k)):
        HELPERS[_k] = getattr(statistics, _k)
HELPERS['avg'] = statistics.fmean


def namespace(names):
    """ The helpers among the given names. """
    return {n: HELPERS[n] for n in names if n in HELPERS}


def _call(expr, ns):
    try:
        return eval(expr, dict(ns))
    except ZeroDivisionError:
        return None


def compute(definitions, raw, lazy_data, helpers, wanted=None):
    """ Compute the features (see the module's docstring).

    :param definitions: {name: {'expr': code object or string, 'boolean': bool, 'fail': str, 'raw': bool}} where
                         'raw' tells if the expression only needs raw extracted data (first pass)
    :param raw:         raw extracted data, e.g. {'pefeats': [...]}
    :param lazy_data:   function returning the data of the second pass, e.g. {'binary': ..., 'pe': ..., constants}
    :param helpers:     evaluation helpers (see namespace)
    :param wanted:      names to return (default: all the computed features)
    """
    values = {}
    # first pass: features from raw data only
    for name, d in definitions.items():
        if d['raw']:
            v = _call(d['expr'], {**helpers, **raw})
            values[name] = bool(v) if d['boolean'] else v
    # second pass: the other features, dependencies first
    todo = [n for n, d in definitions.items() if not d['raw']]
    if todo:
        data, undefined = {**helpers, **lazy_data(), **raw}, set()
        def _compute(name, stack=()):
            if name in values or name in undefined:
                return
            if name in stack:
                raise ExtractionError(f"circular dependency of '{name}'")
            d = definitions[name]
            for dep in d.get('deps', []):
                _compute(dep, stack + (name, ))
                if dep in undefined:   # pbox keeps on retrying, then fails ("Too much iterations")
                    raise ExtractionError(f"'{name}' depends on '{dep}', which could not be computed")
            try:
                v = _call(d['expr'], {**data, **values})
            except Exception as e:
                if d['fail'] == "continue":
                    v = None
                elif d['fail'] == "warning":
                    undefined.add(name)
                    return
                else:
                    raise ExtractionError(f"{name}: {e.__class__.__name__}: {e}") from None
            values[name] = bool(v) if d['boolean'] else v
        for name in todo:
            _compute(name)
    return values if wanted is None else {n: values.get(n) for n in wanted}
