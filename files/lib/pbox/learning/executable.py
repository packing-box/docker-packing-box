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
        fields = Executable.FIELDS + ["hash", "label", "Index"]
        self = super(Executable, cls).__new__(cls, *parts, **kwargs)
        self._selection = {'pre': [], 'post': []}
        if hasattr(self, "_dataset"):
            try:
                d = self._dataset._data[self._dataset._data.hash == self.hash].iloc[0].to_dict()
                f = {a: v for a, v in d.items() if a not in fields if str(v) != "nan"}
                if len(f) > 0:
                    setattr(self, "data", f)
                    self.selection = list(f.keys())
            except AttributeError:
                pass  # this occurs when the dataset is still empty, therefore holding no 'hash' column
            except IndexError:
                pass  # this occurs when the executable did not exist in the dataset yet
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
        l = self._selection['post']
        return data if len(l) == 0 else {n: f for n, f in data.items() if n in l}
    
    @property
    def features(self):
        return {n: FEATURE_DESCRIPTIONS.get(n, "") for n in self.data.keys()}
    
    @property
    def selection(self):
        l = self._selection['pre']
        return self._features.copy() if len(l) == 0 else {n: f for n, f in self._features.items() if n in l}
    
    @selection.setter
    def selection(self, features):
        if features is None:
            return
        if not isinstance(features, (list, tuple)):
            features = [features]
        if isinstance(features, (list, tuple)):
            for f in features:
                # this will filter groups of features before computation (e.g. "pefeats")
                if re.match(r"\{.*\}$", f):
                    self._selection['pre'].append(f[1:-1])
                # this will filter features ones computed (e.g. "dll_characteristics_...", part of the "pefeats" group)
                else:
                    self._selection['post'].append(f)
        try:
            del self.data
        except AttributeError:
            pass

