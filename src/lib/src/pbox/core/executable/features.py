# -*- coding: UTF-8 -*-
from collections import deque
from tinyscript import itertools, logging, re
from tinyscript.helpers import is_generator as is_gen, Path

from ...helpers import dict2, expand_formats, get_data, load_yaml_config, MetaBase

lazy_load_module("yaml")


__all__ = ["Features"]


class Feature(dict2):
    def __call__(self, data, *args, **kwargs):
        self._exe = data.get('executable')
        try:
            return super().__call__(data, *args, **kwargs)
        except ZeroDivisionError:  # i.e. when a ratio has its denominator set to 0 ;
            return                 #  in this case, feature's value is undefined
    
    @cached_property
    def boolean(self):
        return any(self.name.startswith(p) for p in ["is_", "has_"])
    
    @cached_property
    def dependencies(self):
        return list(set([x for x in re.split(r"[\s\.\,\-\+\[\]\(\)]", self.result or "") if x in Features]))
    
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
        return self.get('parser', config[f'{p}_parser'])


class Features(dict, metaclass=MetaBase):
    """ This class parses the YAML definitions of features to be derived from the extracted ones.
    
    NB: On the contrary of abstractions (e.g. Packer, Detector), Features lazily computes its registry.
    """
    boolean_only = False
    registry     = None
    
    def __init__(self, exe=None):
        ft, l = Features, self.__class__.logger
        # parse YAML features definition once
        if ft.registry is None:
            src = ft.source  # WARNING! this line must appear BEFORE ft.registry={} because the first time that the
                             #           source attribute is called, it is initialized and the registry is reset to None
            l.debug(f"loading features from {src}...")
            ft.registry = {}
            # important note: the 'keep' parameter is not considered here as some features may be required for computing
            #                  others but not kept in the final data, hence required in the registry yet
            flist = [f for l in [["All"], [f for f in FORMATS.keys() if f != "All"], expand_formats("All")] for f in l]
            for name, params in load_yaml_config(src):
                r, values = params.pop('result', {}), params.pop('values', [])
                # allow to use 'result: ...' instead of 'result:\n  All: ...' to save space
                if not isinstance(r, dict):
                    r = {'All': r}
                # allow to use expressions in the 'values' field
                if isinstance(values, str):
                    values = list(dict2({'result': values})({'get_data': get_data}))
                # consider features for most specific formats first, then intermediate format classes and finally the
                #  collapsed format class "All"
                for fmt in flist:
                    expr = r.get(fmt)
                    if expr is not None:
                        if len(values) > 0:
                            if not all(isinstance(x, (list, set, tuple, range, range2)) or is_gen(x) for x in values):
                                values = [values]
                            f = []
                            for val in itertools.product(*values):
                                p = {k: v for k, v in params.items()}
                                val = val[0] if isinstance(val, tuple) and len(val) == 1 else val
                                try:
                                    e = expr % val
                                except Exception as e:
                                    l.error(f"expression: {expr}")
                                    l.error(f"value:      {val}")
                                    raise
                                try:
                                    n = name % (val.lower() if isinstance(val, str) else val)
                                except TypeError:
                                    l.error(f"name:  {name}")
                                    l.error(f"value: {val}")
                                    raise
                                d = p['description']
                                try:
                                    p['description'] = d % val
                                except TypeError:
                                    l.error(f"description: {name}")
                                    l.error(f"value:       {val}")
                                    raise
                                f.append(Feature(p, name=n, result=e, logger=l))
                        else:
                            f = [Feature(params, name=name, result=expr, logger=l)]
                        for feat in f:
                            for subfmt in expand_formats(fmt):
                                ft.registry.setdefault(subfmt, {})
                                ft.registry[subfmt][feat.name] = feat
            l.debug(f"{len(Features.names())} features loaded")
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
                # set 'binary' as the generic reference for the parsed binary but also for specific formats ('pe', ...)
                d = {'binary': p, exe.group.lower(): p}
                # add some constants
                d.update({c: c for c in FEATURE_CONSTANTS})
                # add raw extracted data
                d.update(self._rawdata)
                # add already computed features
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
                        raise ValueError(f"Too much iterations of '{n}'")
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

