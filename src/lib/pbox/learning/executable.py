# -*- coding: UTF-8 -*-
import re
from contextlib import suppress
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
        self._selection, self.transform = {'pre': [], 'post': []}, kwargs.get("transform")
        if hasattr(self, "_dataset"):
            with suppress(AttributeError, IndexError): # Attr => 'hash' column missing ; Index => exe does not exist yet
                d = self._dataset._data[self._dataset._data.hash == self.hash].iloc[0].to_dict()
                f = {a: v for a, v in d.items() if a not in fields if str(v) != "nan"}
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
        if self._transform is None:
            return self.rawdata
        data, t, _processed = {}, self._transform, []
        # define a private shortcut for setting values
        def __set(n, v):
            if v is not None:
                if isinstance(v, bool):
                    v = int(v)
                data[n] = v
        # define a private function for handling lambda functions in dictionaries
        def __transform(features, values):
            if isinstance(features, tuple) and len(features) == 2:
                features, descr = features
                FEATURE_DESCRIPTIONS[name] = descr
            if isinstance(features, dict):
                for subname, subvalue in features.items():
                    if isinstance(subvalue, tuple) and len(subvalue) == 2:
                        subvalue, descr = subvalue
                        FEATURE_DESCRIPTIONS[subname] = descr
                    if isinstance(subvalue, type(lambda: 0)):
                        subvalue = subvalue(*values)
                    __set(name, subvalue)
            elif isinstance(features, type(lambda: 0)):
                __set(name, features(*values))
        # this processes a class of transformers and afterwards the "default" class (if not already selected),
        #  discarding features whose values are computed to None
        for transform in ([t] + [[], ["default"]][t != "default"]):
            trans = FEATURE_TRANSFORMERS[transform]
            # handle simple features
            for name, value in self.rawdata.items():
                if name in _processed:
                    continue
                if name in trans:
                    _processed.append(name)
                    __transform(trans.get(name, value), (value, ))
                # if no transformer, simply assign the base value
                else:
                    data[name] = value
            # handle combinations
            for combo, features in trans.items():
                if not isinstance(combo, tuple):
                    continue
                name, descr = "_and_".join(combo), "Combination of %s and %s" % (", ".join(combo[:-1]), combo[-1])
                if isinstance(features, type(lambda: 0)):
                    features = {name: (features, descr)}
                __transform(features, (self.rawdata.get(n) for n in combo))
        return data
    
    @property
    def features(self):
        return {n: FEATURE_DESCRIPTIONS.get(n, "") for n in self.data.keys()}
    
    @cached_property
    def rawdata(self):
        data = {}
        for name, func in self.selection.items():
            if name in data.keys():
                continue
            r = func(self)
            if isinstance(r, dict):
                data.update(r)
            else:
                data[name] = r
        l = []
        for f1 in self._selection['post']:
            # handle f1 as exact feature name
            if f1 in data.keys():
                l.append(f1)
                continue
            # handle f1 as pattern
            for f2 in data.keys():
                if re.search(f1, f2, re.I):
                    l.append(f2)
        return data if len(l) == 0 else {n: f for n, f in data.items() if n in l}
    
    @property
    def selection(self):
        l, d = self._selection['pre'], self._features
        return d.copy() if len(l) == 0 else {n: f for n, f in d.items() if n in l}
    
    @selection.setter
    def selection(self, features):
        self._selection = {'pre': [], 'post': []}
        if features is None:
            return
        if not isinstance(features, (list, tuple)):
            features = [features]
        if isinstance(features, list):
            for f in features:
                # this will filter out groups of features before computation (e.g. "pefeats")
                if re.match(r"\{.*\}$", f):
                    self._selection['pre'].append(f[1:-1])
                # this will filter features once computed (e.g. "dll_characteristics_...", part of the "pefeats" group)
                else:
                    self._selection['post'].append(f)
        if hasattr(self, "rawdata"):
            if hasattr(self, "data"):
                del self.data
            # if all new features are included in the former data, just filter out
            #  NB: by design, the following if condition does not handle patterns, meaning that when using them,
            #       self.rawdata will be deleted ; this is logical if we consider that patterns may match only a part of 
            #       already existing data and then provide an incomplete result
            if all(f in self.rawdata.keys() for f in self._selection['post']):
                self.rawdata = {k: v for k, v in self.rawdata.items() if k in self._selection['post']}
            # otherwise, reset the data attribute so that it is lazily recomputed at next call
            else:
                del self.rawdata
    
    @property
    def transform(self):
        return self._transform
    
    @transform.setter
    def transform(self, value):
        if value is not None and value not in FEATURE_TRANSFORMERS.keys():
            raise ValueError("Bad transformer ; shall be one of: " % "|".join(FEATURE_TRANSFORMERS.keys()))
        self._transform = value

