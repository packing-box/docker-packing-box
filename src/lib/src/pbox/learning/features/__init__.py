# -*- coding: UTF-8 -*-
from collections import deque
from functools import cached_property
from tinyscript import logging, re
from tinyscript.helpers import lazy_load_module, Path

from ...common.config import config
from ...common.utils import dict2, expand_formats, FORMATS

lazy_load_module("yaml")


__all__ = ["Features"]


class Feature(dict2):
    _fields = {'keep': True, 'values': []}  # default values that will be set in dict2.__init__

    def __init__(self, idict, **kwargs):
        super(Feature, self).__init__(idict, **kwargs)
        self['boolean'] = self.__dict__['boolean'] = any(self['name'].startswith(p) for p in ["is_", "has_"])
    
    @cached_property
    def dependencies(self):
        return list(set([x for x in re.split(r"[\s\.\[\]\(\)]", self.result or "") if x in Features]))


class MetaFeatures(type):
    """ This metaclass allows to iterate feature names over the class-bound registry of features. """
    def __iter__(self):
        Features(None)  # trigger registry's lazy initialization
        temp = []
        for features in self.registry.values():
            for name in features.keys():
                if name not in temp:
                    yield name
                    temp.append(name)

    @property
    def source(self):
        if not hasattr(self, "_source"):
            self.source = None  # use the default source from 'config'
        return self._source

    @source.setter
    def source(self, path):
        p = Path(str(path or config['features']), expand=True)
        if hasattr(self, "_source") and self._source == p:
            return
        self._source = p
        self.registry = None  # reset the registry


class Features(dict, metaclass=MetaFeatures):
    """ This class parses the YAML definitions of features to be derived from the extracted ones.
    
    NB: On the contrary of abstractions (e.g. Packer, Detector), Features lazily computes its registry.
    """
    boolean_only = False
    registry     = None
    
    @logging.bindLogger
    def __init__(self, exe):
        # parse YAML features definition once
        if Features.registry is None:
            # open the target YAML-formatted features set only once
            with Features.source.open() as f:
                features = yaml.load(f, Loader=yaml.Loader) or {}
            Features.registry = {}
            # collect properties that are applicable for all the other features
            data_all = features.pop('ALL', {})
            # important note: the 'keep' parameter is not considered here as some features may be required for computing
            #                  others but not kept in the final data, hence required in the registry yet
            for name, params in features.items():
                for i in data_all.items():
                    params.setdefault(*i)
                r, v = params.pop('result', {}), params.pop('values', [])
                # consider features for most specific formats first, then intermediate format classes and finally the
                #  collapsed format class "All"
                for flist in [expand_formats("All"), [f for f in FORMATS.keys() if f != "All"], ["All"]]:
                    for fmt in flist:
                        expr = r.get(fmt)
                        if expr:
                            if len(v) > 0:
                                f = []
                                for val in v:
                                    p = {k: v for k, v in params.items()}
                                    #TODO: support dynamic free variables (not only "x")
                                    e = expr.replace("x", str(val))
                                    try:
                                        n = name % val
                                    except TypeError:
                                        self.logger.error("missing formatter in name '%s'" % d)
                                        raise
                                    d = p.pop("description")
                                    try:
                                        p['description'] = d % val
                                    except TypeError:
                                        self.logger.warning("name: %s" % n)
                                        self.logger.error("missing formatter in description '%s'" % d)
                                        raise
                                    f.append(Feature(p, name=n, parent=self, result=e))
                            else:
                                f = [Feature(params, name=name, parent=self, result=expr)]
                            for feat in f:
                                for subfmt in expand_formats(fmt):
                                    Features.registry.setdefault(subfmt, {})
                                    Features.registry[subfmt][feat.name] = feat
        if exe is not None and exe.format in Features.registry:
            from .extractors import Extractors
            self._rawdata = Extractors(exe)
            todo, counts, reg = deque(), {}, Features.registry[exe.format]
            # compute features based on the extracted values first
            for name, feature in reg.items():
                # compute only if it has the keep=True flag ; otherwise, it will be lazily computed on need
                if (not Features.boolean_only or Features.boolean_only and feature.boolean) and feature.keep:
                    try:
                        self[name] = feature(self._rawdata, True)
                    except NameError:
                        todo.append(feature)
                    except ValueError:  # occurs when FobiddenNodeError is thrown
                        continue
            # then lazily compute features until we converge in a state where all the required features are computed
            while len(todo) > 0:
                feature = todo.popleft()
                n = feature.name
                d = {}
                d.update(self._rawdata)
                d.update(self)
                try:
                    self[n] = feature(d)
                except NameError:
                    bad = False
                    # every feature dependency has already been seen, but yet feature computation fails
                    if all(name2 in counts for name2 in feature.dependencies):
                        counts.setdefault(n, 0)
                        counts[n] += 1
                    else:
                        for name2 in feature.dependencies:
                            if name2 not in reg:
                                del reg[n]
                                if n in counts:
                                    del counts[n]
                                bad = True
                                break
                        if not bad:
                            for name2 in feature.dependencies:
                                # compute the dependency in priority
                                todo.appendleft(reg[name2])
                            counts.setdefault(name2, 0)
                    if counts.get(n, 0) > 10:
                        raise ValueError("Too much iterations of '%s'" % n)
                except ValueError:  # occurs when FobiddenNodeError is thrown
                    continue
            # once converged, ensure that we did not leave a feature that should not be kept
            do_not_keep = []
            for name in self:
                if not reg[name].keep:
                    do_not_keep.append(name)
            for name in do_not_keep:
                del self[name]
    
    def __getitem__(self, name):
        value = super(Features, self).__getitem__(name)
        # if string, this may be a flat list/dictionary converted for working with pandas.DataFrame (cfr error:
        #  ValueError: Must have equal len keys and value when setting with an iterable)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
        return value
    
    @staticmethod
    def names(format="All"):
        Features(None)  # force registry initialization
        l = []
        for c in expand_formats(format):
            l.extend(list(Features.registry[c].keys()))
        return sorted(list(set(l)))

