# -*- coding: UTF-8 -*-
import sklearn.utils._param_validation as pv
from numbers import Number
from sklearn.utils._param_validation import Options
from sklearn.utils.validation import _is_arraylike_not_scalar


class _IterableOfLengthN(Options):
    """Constraint representing array-likes of predefined length N."""
    def __init__(self, n, type=Number):
        self.hidden = False
        self.n = n
        self.type = type
        self._check_params()
    
    def _check_params(self):
        if self.type not in [Number, str]:
            raise ValueError(f"type must be either Number or str. Got {self.type} instead.")
        if not isinstance(self.n, int) and self.n > 0:
            raise ValueError(f"n must be a non-null positive integer. Got {self.n} instead.")


class ArrayOfLengthN(_IterableOfLengthN):
    """Constraint representing array-likes of predefined length N."""
    def is_satisfied_by(self, val):
        return _is_arraylike_not_scalar(val) and len(val) == self.n and all(isinstance(v, self.type) for v in val)
    
    def __str__(self):
        return f"a array-like of numbers or strings with a predefined length of {self.n}"
pv.ArrayOfLengthN = ArrayOfLengthN


class DictOfLengthAtLeastN(_IterableOfLengthN):
    """Constraint representing a dictionary of predefined length N."""
    def is_satisfied_by(self, val):
        return isinstance(val, dict) and len(val) >= self.n and all(isinstance(v, self.type) for v in val.values())
    
    def __str__(self):
        return f"a dictionary of numbers or strings with a predefined minimal length of {self.n}"
pv.DictOfLengthAtLeastN = DictOfLengthAtLeastN


class IntOptions(Options):
    """Constraint representing a finite set of integers.

    Parameters
    ----------
    options : set of int
        The set of valid integers.

    deprecated : set of int or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """
    def __init__(self, options, *, deprecated=None):
        super().__init__(type=int, options=options, deprecated=deprecated)
pv.IntOptions = IntOptions

