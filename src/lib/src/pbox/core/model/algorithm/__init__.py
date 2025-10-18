# -*- coding: UTF-8 -*-
from tinyscript.helpers import Path

from ....helpers.items import load_yaml_config


__all__ = ["Algorithm"]

__cls = None
__initialized = False


def __init_metaalgo():
    global Algorithm
    from ....helpers.items import MetaItem  # this imports the lazy object Proxy (with its specific id(...))
    MetaItem.__name__                       # this forces the initialization of the Proxy
    from ....helpers.items import MetaItem  # this reimports the loaded metaclass (hence, getting the right id(...))
    
    class MetaAlgorithm(MetaItem):
        def __getattribute__(self, name):
            # this masks some attributes for child classes (e.g. Algorithm.registry can be accessed, but when the
            #  registry of child classes is computed, the child classes, e.g. RF, won't be able to access RF.registry)
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError(f"'{self.__name__}' object has no attribute '{name}'")
            return super(MetaAlgorithm, self).__getattribute__(name)
        
        @property
        def config(self):
            if not hasattr(self, "_config"):
                self.config = None
            return self._config
        
        @config.setter
        def config(self, path):
            p = Path(str(path or config['algorithms']), expand=True)
            if hasattr(self, "_config") and self._config == p:
                return
            Algorithm.__name__  # force initialization
            cls, self._config, glob = Algorithm, p, globals()
            # remove the child classes of the former registry from the global scope
            for child in getattr(self, "registry", []):
                glob.pop(child.cname, None)
            cls.registry = []
            # start parsing items of cls
            _labellings = {'supervised': "full", 'semi-supervised': "partial", 'unsupervised': "none", \
                           'heuristics': "full"}
            algos = {k: v for k, v in load_yaml_config(p, ["base"])}
            # backward compatibility with former algorithms.yml format
            if all(k.lower() in _labellings for k in algos.keys()):
                d = {}
                for category, items in algos.items():
                    dflts = items.pop('defaults', {})
                    dflts['category'] = category.lower()
                    for algo, data in items.items():
                        for k, v in dflts.items():
                            if k == "base":
                                raise ValueError("parameter 'base' cannot have a default value")
                            data.setdefault(k, v)
                        d[algo] = data
                algos = d
            for algo, data in algos.items():
                if data.get('category') not in _labellings.keys():
                    raise ValueError(f"bad learning algorithm category ({data.get('category')})")
                data.setdefault('boolean', False)
                data.setdefault('multiclass', True)
                data.setdefault('parameters', {})
                data['labelling'] = _labellings[data['category']]
                # put the related algorithm in module's globals()
                d = dict(cls.__dict__)
                for a in ["get", "iteritems", "mro", "registry"]:
                    d.pop(a, None)
                i = glob[algo] = type(algo, (cls, ), d)
                i._instantiable = True
                # now set attributes from YAML parameters
                for k, v in data.items():
                    setattr(i, k, v)
                glob['__all__'].append(algo)
                cls.registry.append(i())
    return MetaAlgorithm
lazy_load_object("MetaAlgorithm", __init_metaalgo)


def __init_algo():
    global __cls, __initialized, MetaAlgorithm
    from ....helpers.items import Item  # this imports the lazy object Proxy (with its specific id(...))
    Item.__name__                       # this forces the initialization of the Proxy
    from ....helpers.items import Item  # this reimports the loaded metaclass (hence, getting the right id(...))
    MetaAlgorithm.__name__
    
    class Algorithm(Item, metaclass=MetaAlgorithm):
        """ Algorithm abstraction. """
        def __getattribute__(self, name):
            # this masks some attributes for child instances in the same way as for child classes
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            return super(Algorithm, self).__getattribute__(name)
        
        def is_weka(self):
            """ Simple method for checking if the algorithm is based on a Weka class. """
            from .weka import WekaClassifier
            return self.base.__base__ is WekaClassifier
        
        @property
        def config(self):
            if not hasattr(self, "_config"):
                self.__class__._config = ""
            return self.__class__._config
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # initialize the registry of algorithms from the default config (~/.packing-box/conf/algorithms.yml)
        Algorithm.config = None  # needs to be initialized, i.e. for the 'model' tool as the registry is used for
    if __cls:                    #  choices, even though the relying YAML config can be tuned via --algorithms-set
        return __cls
    __cls = Algorithm
    return Algorithm
lazy_load_object("Algorithm", __init_algo)

