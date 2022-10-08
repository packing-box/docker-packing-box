# -*- coding: UTF-8 -*-
import yaml
from collections import deque
from functools import cached_property
from tinyscript import logging, re

from .extractors import Extractors
from ...common.config import config
from ...common.utils import dict2, expand_formats, FORMATS


__all__ = ["Features"]


class Feature(dict2):
    _fields = {'keep': True, 'values': []}

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


class Features(dict, metaclass=MetaFeatures):
    """ This class parses the YAML definitions of features to be derived from the extracted ones. """
    boolean_only = False
    registry     = None
    source       = config['features']
    
    @logging.bindLogger
    def __init__(self, exe):
        # parse YAML features definition once
        if Features.registry is None:
            # open the target YAML-formatted features set only once
            with open(Features.source) as f:
                features = yaml.load(f, Loader=yaml.Loader) or {}
            Features.registry = {}
            # collect properties that are applicable for all the other features
            data_all = features.pop('ALL', {})
            for name, params in features.items():
                for i in data_all.items():
                    params.setdefault(*i)
                r = params.pop('result', {})
                v = params.pop('values', [])
                # consider most specific features first, then intermediate classes and finally the collapsed class "All"
                for clist in [expand_formats("All"), list(FORMATS.keys())[1:], ["All"]]:
                    for c in clist:
                        expr = r.get(c)
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
                                for c2 in expand_formats(c):
                                    Features.registry.setdefault(c2, {})
                                    Features.registry[c2][feat.name] = feat
        if exe is not None:
            self._rawdata = Extractors(exe)
            todo, counts = deque(), {}
            # compute features based on the extracted values first
            for name, feature in Features.registry[exe.format].items():
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
                for ns in [_EVAL_NAMESPACE, self._rawdata, self]:
                    d.update(ns)
                try:
                    self[n] = feature(d)
                except NameError:
                    # every feature dependency has already been seen, but yet feature computation fails
                    if all(name2 in counts for name2 in feature.dependencies):
                        counts.setdefault(n, 0)
                        counts[n] += 1
                    else:
                        for name2 in feature.dependencies:
                            # compute the dependency in priority
                            todo.appendleft(Features.registry[exe.format][name2])
                            counts.setdefault(name2, 0)
                    if counts[n] > 10:
                        raise ValueError("Too much iterations of '%s'" % n)
                except ValueError:  # occurs when FobiddenNodeError is thrown
                    continue
            # once converged, ensure that we did not leave a feature that should be kept
            for name in self:
                if not Features.registry[exe.format][name].keep:
                    self.pop(name, None)
    
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

