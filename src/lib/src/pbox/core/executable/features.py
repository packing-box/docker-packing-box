# -*- coding: UTF-8 -*-
from collections import deque
from tinyscript import itertools, logging, re
from tinyscript.helpers import is_generator, lazy_load_module, Path

from ...helpers import dict2, expand_formats, load_yaml_config, MetaBase, FORMATS

lazy_load_module("yaml")


__all__ = ["Features"]


class Feature(dict2):
    def __call__(self, data, *args, **kwargs):
        self._exe = data.get('executable')
        return super().__call__(data, *args, **kwargs)
    
    @cached_property
    def boolean(self):
        return any(self.name.startswith(p) for p in ["is_", "has_"])
    
    @cached_property
    def dependencies(self):
        return list(set([x for x in re.split(r"[\s\.\[\]\(\)]", self.result or "") if x in Features]))
    
    # default value on purpose not set via self.setdefault(...)
    @cached_property
    def keep(self):
        return self.get('keep', True)
    
    # 'parser' parameter in the YAML config has precedence on the globally configured parser
    @cached_property
    def parser(self):
        try:
            p = self._exe.shortgroup
            delattr(self, "_exe")
        except AttributeError:
            p = "default"
        return self.get('parser', config['%s_parser' % p])


class Features(dict, metaclass=MetaBase):
    """ This class parses the YAML definitions of features to be derived from the extracted ones.
    
    NB: On the contrary of abstractions (e.g. Packer, Detector), Features lazily computes its registry.
    """
    boolean_only = False
    registry     = None
    
    @logging.bindLogger
    def __init__(self, exe=None):
        ft = Features
        # parse YAML features definition once
        if ft.registry is None:
            src = ft.source  # WARNING! this line must appear BEFORE ft.registry={} because the first time that the
                             #           source attribute is called, it is initialized and the registry is reset to None
            self.logger.debug("loading features from %s..." % src)
            ft.registry = {}
            # important note: the 'keep' parameter is not considered here as some features may be required for computing
            #                  others but not kept in the final data, hence required in the registry yet
            flist = [f for l in [["All"], [f for f in FORMATS.keys() if f != "All"], expand_formats("All")] for f in l]
            for name, params in load_yaml_config(src):
                r, values = params.pop('result', {}), params.pop('values', [])
                # consider features for most specific formats first, then intermediate format classes and finally the
                #  collapsed format class "All"
                for fmt in flist:
                    expr = r.get(fmt)
                    if expr is not None:
                        if len(values) > 0:
                            if not all(isinstance(x, (list, tuple)) or is_generator(x) for x in values):
                                values = [values]
                            f = []
                            for val in itertools.product(*values):
                                p = {k: v for k, v in params.items()}
                                try:
                                    e = expr % val
                                except Exception as e:
                                    self.logger.error("expression: %s" % expr)
                                    self.logger.error("value:      %s (%s)" % (str(val), type(val)))
                                    raise
                                try:
                                    n = name % val
                                except TypeError:
                                    self.logger.error("missing formatter in name '%s'" % d)
                                    raise
                                d = p['description']
                                try:
                                    p['description'] = d % val
                                except TypeError:
                                    self.logger.warning("name: %s" % n)
                                    self.logger.error("missing formatter in description '%s'" % d)
                                    raise
                                f.append(Feature(p, name=n, result=e, logger=self.logger))
                        else:
                            f = [Feature(params, name=name, result=expr, logger=self.logger)]
                        for feat in f:
                            for subfmt in expand_formats(fmt):
                                ft.registry.setdefault(subfmt, {})
                                ft.registry[subfmt][feat.name] = feat
        if exe is not None and exe.format in ft.registry:
            from .extractors import Extractors
            self._rawdata = Extractors(exe)
            todo, counts, reg = deque(), {}, ft.registry[exe.format]
            # compute features based on the extracted values first
            for name, feature in reg.items():
                # compute only if it has the keep=True flag ; otherwise, it will be lazily computed on need
                if (not ft.boolean_only or ft.boolean_only and feature.boolean) and feature.keep:
                    try:
                        v = feature(self._rawdata, True)
                        self[name] = bool(v) if feature.boolean else v
                    except NameError:
                        todo.append(feature)
                    except ValueError:  # occurs when FobiddenNodeError is thrown
                        continue
            # then lazily compute features until we converge in a state where all the required features are computed
            while len(todo) > 0:
                feature = todo.popleft()
                n = feature.name
                p = exe.parse(feature.parser, reset=False)
                d = {'binary': p, exe.group.lower(): p}
                d.update(self._rawdata)
                d.update(self)
                try:
                    v = feature(d)
                    self[n] = bool(v) if feature.boolean else v
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
                return literal_eval(value)
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

