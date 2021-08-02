# -*- coding: UTF-8 -*-
import re
from functools import cached_property

from .features import *
from ..common.executable import Executable as Base


__all__ = ["Executable"]


class Executable(Base):
    """ Executable extension for handling features. """
    _features = {}
    
    def __new__(cls, *parts, **kwargs):
        self = super(Executable, cls).__new__(cls, *parts, **kwargs)
        self._selection = None
        if hasattr(self, "_dataset"):
            h = kwargs.pop('hash', self.basename)
            d = self._dataset._data[self._dataset._data.hash == h].iloc[0].to_dict()
            f = {a: v for a, v in d.items() if a not in Executable.FIELDS + ["hash", "label"]}
            if len(f) > 0:
                setattr(self, "data", f)
                self.selection = list(f.keys())
        return self
    
    def __getattribute__(self, name):
        if name == "_features":
            # populate Executable._features with the relevant set of features and their related functions
            fset = Executable._features.get(self.category)
            if fset is None:
                Executable._features[self.category] = fset = Features(self.category)
            return fset
        return super(Executable, self).__getattribute__(name)
    
    @cached_property
    def data(self):
        data = {}
        for name, func in self.selection.items():
            r = func(self)
            if isinstance(r, dict):
                data.update(r)
            else:
                data[name] = r
        return data
    
    @property
    def features(self):
        return {n: FEATURE_DESCRIPTIONS.get(n, "") for n in self.data.keys()}
    
    @property
    def selection(self):
        return {n: f for n, f in self._features.items() if n in (self._selection or self._features.keys())}
    
    @selection.setter
    def selection(self, features):
        if isinstance(features, (list, tuple)):
            self._selection = features
        elif isinstance(features, str):
            self._selection = [x for x in self.features.keys() if re.search(features, x)]

