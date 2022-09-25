# -*- coding: UTF-8 -*-
import builtins
import yaml
from ast import literal_eval
from tinyscript import logging
from tinyscript.helpers.expressions import WL_NODES  # note: eval2 is bound to the builtins, hence not imported

from .extractors import Extractors
from ...common.config import config
from ...common.utils import expand_formats, FORMATS


__all__ = ["Features"]


_EVAL_NAMESPACE = {k: getattr(builtins, k) for k in ["abs", "divmod", "float", "hash", "hex", "id", "int", "len",
                                                     "list", "max", "min", "oct", "ord", "pow", "range", "round", "set",
                                                     "str", "sum", "tuple", "type"]}
WL_EXTRA_NODES = ("arg", "arguments", "keyword", "lambda")


class Feature(dict):
    def __init__(self, idict, **kwargs):
        self.setdefault("name", "undefined")
        self.setdefault("description", "")
        self.setdefault("result", None)
        super(Feature, self).__init__(idict, **kwargs)
        self['boolean'] = any(self['name'].startswith(p) for p in ["is_", "has_"])
        self.__dict__ = self
        if self.result is None:
            raise ValueError("%s: 'result' shall be defined" % self.name)
    
    def __call__(self, data, silent=False):
        try:
            return eval2(self.result, _EVAL_NAMESPACE, data, whitelist_nodes=WL_NODES + WL_EXTRA_NODES)
        except:
            if not silent:
                self.parent.logger.warning("Bad expression: %s" % self.result)
                self.parent.logger.debug("Variables:\n- %s" % \
                                         "\n- ".join("%s(%s)=%s" % (k, type(v).__name__, v) for k, v in data.items()))
            raise


class Features(dict):
    """ This class parses the YAML definitions of features to be derived from the extracted ones. """
    boolean_only = False
    registry     = None
    source       = config['features']
    
    @logging.bindLogger
    def __init__(self, exe):
        self._rawdata = Extractors(exe)
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
                v = params.pop('values', {})
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
            todo = {}
            # compute features based on the extracted values first
            for name, feature in Features.registry[exe.format].items():
                if not Features.boolean_only or Features.boolean_only and feature.boolean:
                    try:
                        self[name] = feature(self._rawdata, True)
                    except NameError:
                        todo[name] = 1
            # then process what could not be computed yet until every feature has a value (eventually based on newly
            #  computed features)
            while len(todo) > 0:
                for name in list(todo.keys()):
                    n = todo.pop(name, 1)
                    try:
                        self[name] = Features.registry[exe.format][name](self)
                    except NameError:
                        if n > 10:
                            raise ValueError("Too much iterations of '%s'" % name)
                        todo[name] = n + 1
    
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

