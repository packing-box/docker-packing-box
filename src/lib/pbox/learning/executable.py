# -*- coding: UTF-8 -*-
import re
from contextlib import suppress
from functools import cached_property

from .features import *
from ..common.executable import Executable as Base


__all__ = ["Executable"]


class Executable(Base):
    """ Executable extension for handling features. """
    _boolean_only  = False
    _features      = {}
    _metadata_only = False
    _source        = "/opt/features.yml"
    _transformers  = None
    
    def __new__(cls, *parts, **kwargs):
        self = super(Executable, cls).__new__(cls, *parts, **kwargs)
        self._selection, self.transform = {'pre': [], 'post': []}, kwargs.get("transform")
        # i.e. dataset commands "show" and "view" only require the executable's metadata, not the data and rawdata
        if not cls._metadata_only:
            self.__getdata()
        return self
    
    def __delattr_(self, name):
        if name in ["rawdata", "data"] and hasattr(self, "_dataset") and not self._dataset._files:
            self.__getdata()
    
    def __getattribute__(self, name):
        if name == "_features":
            # populate Executable._features with the relevant set of features and their related functions
            eset = Executable._features.get(self.format)
            if eset is None:
                Executable._features[self.format] = eset = Extractors(self.format)
            return eset
        if name == "_transformers":
            tset = Executable._transformers
            if tset is None:
                # populate Executable._transformers with the set of feature transformers and their related functions
                Executable._transformers = tset = Transformers(self._features, self._boolean_only, self._source)
            return tset
        return super(Executable, self).__getattribute__(name)
    
    def __getdata(self):
        if hasattr(self, "_dataset"):
            fields = Executable.FIELDS + ["hash", "label", "Index"]
            with suppress(AttributeError, IndexError): # Attr => 'hash' column missing ; Index => exe does not exist yet
                d = self._dataset._data[self._dataset._data.hash == self.hash].iloc[0].to_dict()
                f = {a: v for a, v in d.items() if a not in fields if str(v) != "nan"}
                if len(f) > 0:
                    setattr(self, "rawdata", f)
                    self.selection = list(f.keys())
    
    @cached_property
    def data(self):
        data, done, tbd = {k: v for k, v in self.rawdata.items()}, [], {}
        def _new_features():
            new_features = feature(self.rawdata)
            for name2 in list(new_features.keys()):
                if name2 in data.keys():
                    tbd[name2] = new_features.pop(name2)
            data.update(new_features)
            done.append(name)
        # compute all derived features (leaving every raw feature as is)
        for name, feature in self._transformers.items():
            if name not in data.keys():
                _new_features()
        # overwrite or remove raw features
        for name, feature in self._transformers.items():
            if name not in done:
                _new_features()
        # process the to-be-done computed features that may have clashed with existing values
        for name, value in tbd.items():
            data[name] = value
        # finally, only keep the boolean features if relevant
        if self._boolean_only:
            data = {k: v for k, v in data.items() if v.boolean}
        return data
    
    @property
    def features(self):
        return {n: FEATURE_DESCRIPTIONS.get(n, "") for n in self.data.keys()}
    
    @cached_property
    def rawdata(self):
        data = Features()
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
        return data if len(l) == 0 else Features({n: f for n, f in data.items() if n in l})
    
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
                self.rawdata = Features({k: v for k, v in self.rawdata.items() if k in self._selection['post']})
            # otherwise, reset the data attribute so that it is lazily recomputed at next call
            else:
                del self.rawdata

