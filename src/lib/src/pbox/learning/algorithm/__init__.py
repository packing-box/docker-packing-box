# -*- coding: UTF-8 -*-
from tinyscript.helpers import lazy_load_module, lazy_load_object, Path, TempPath

lazy_load_module("yaml")


__all__ = ["Algorithm"]

__cls = None
__initialized = False


_sanitize_feature_name = lambda n: n.replace("<", "[lt]").replace(">", "[gt]")


def __init_metaalgo():
    global Algorithm
    from ...common.config import config
    from ...common.item import MetaItem  # this imports the lazy object Proxy (with its specific id(...))
    MetaItem.__name__                   # this forces the initialization of the Proxy
    from ...common.item import MetaItem  # this reimports the loaded metaclass (hence, getting the right id(...))
    
    class MetaAlgorithm(MetaItem):
        def __getattribute__(self, name):
            # this masks some attributes for child classes (e.g. Algorithm.registry can be accessed, but when the
            #  registry of child classes is computed, the child classes, e.g. RF, won't be able to access RF.registry)
            if name in ["get", "iteritems", "mro", "registry", "source"] and self._instantiable:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__name__, name))
            return super(MetaAlgorithm, self).__getattribute__(name)
        
        @property
        def source(self):
            return self._source

        @source.setter
        def source(self, path):
            p = Path(str(path or config['algorithms']), expand=True)
            if hasattr(self, "_source") and self._source == p:
                return
            Algorithm.__name__  # force initialization
            cls, self._source, glob = Algorithm, p, globals()
            # remove the child classes of the former registry from the global scope
            for child in getattr(self, "registry", []):
                glob.pop(child.cname, None)
            # open the .conf file associated to algorithms
            cls.registry = []
            with p.open() as f:
                algos = yaml.load(f, Loader=yaml.Loader)
            # start parsing items of cls
            for category, items in algos.items():
                if category not in ["Semi-Supervised", "Supervised", "Unsupervised"]:
                    raise ValueError("bad learning algorithm category (%s)" % category)
                dflts = items.pop('defaults', {})
                dflts.setdefault('boolean', False)
                dflts.setdefault('multiclass', True)
                dflts.setdefault('parameters', {})
                dflts['labelling'] = {'Supervised': "full", 'Semi-Supervised': "partial", 'Unsupervised': "none"} \
                                     [category]
                for algo, data in items.items():
                    for k, v in dflts.items():
                        if k == "base":
                            raise ValueError("parameter 'base' cannot have a default value")
                        data.setdefault(k, v)
                    # put the related algorithm in module's globals()
                    d = dict(cls.__dict__)
                    for a in ["get", "iteritems", "mro", "registry"]:
                        d.pop(a, None)
                    i = glob[algo] = type(algo, (cls, ), d)
                    i._instantiable = True
                    # now set attributes from YAML parameters
                    for k, v in data.items():
                        setattr(i, "_" + k if k == "source" else k, v)
                    glob['__all__'].append(algo)
                    cls.registry.append(i())
    return MetaAlgorithm
lazy_load_object("MetaAlgorithm", __init_metaalgo)


def __init_algo():
    global __cls, __initialized, MetaAlgorithm
    from ...common.item import Item  # this imports the lazy object Proxy (with its specific id(...))
    Item.__name__                   # this forces the initialization of the Proxy
    from ...common.item import Item  # this reimports the loaded metaclass (hence, getting the right id(...))
    MetaAlgorithm.__name__
    
    class Algorithm(Item, metaclass=MetaAlgorithm):
        """ Algorithm abstraction. """
        def __getattribute__(self, name):
            # this masks some attributes for child instances in the same way as for child classes
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
            return super(Item, self).__getattribute__(name)
        
        def is_weka(self):
            """ Simple method for checking if the algorithm is based on a Weka class. """
            from .weka import WekaClassifier
            return self.base.__base__ is WekaClassifier
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # initialize the registry of algorithms from the default source (~/.opt/algorithms.yml)
        Algorithm.source = None
    if __cls:
        return __cls
    __cls = Algorithm
    return Algorithm
lazy_load_object("Algorithm", __init_algo)

