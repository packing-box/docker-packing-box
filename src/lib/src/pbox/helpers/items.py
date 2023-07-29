# -*- coding: UTF-8 -*-
import builtins
from tinyscript import inspect, random, re
from tinyscript.helpers import is_file, is_folder, lazy_load_module, lazy_object, Path, TempPath
from tinyscript.helpers.expressions import WL_NODES

from ..core.config import *

lazy_load_module("yaml")


__all__ = ["backup", "dict2", "make_registry", "purge_items", "select_features", "MetaBase"]

_EVAL_NAMESPACE = {k: getattr(builtins, k) for k in ["abs", "divmod", "float", "hash", "hex", "id", "int", "len",
                                                     "list", "max", "min", "next", "oct", "ord", "pow", "range",
                                                     "range2", "round", "set", "str", "sum", "tuple", "type"]}
WL_EXTRA_NODES = ("arg", "arguments", "keyword", "lambda")


class dict2(dict):
    """ Simple extension of dict for defining callable items. """
    def __init__(self, idict, **kwargs):
        self.setdefault("name", "undefined")
        self.setdefault("description", "")
        self.setdefault("result", None)
        self.setdefault("parameters", {})
        for f, v in getattr(self.__class__, "_fields", {}).items():
            self.setdefault(f, v)
        super().__init__(idict, **kwargs)
        self.__dict__ = self
        if self.result is None:
            raise ValueError("%s: 'result' shall be defined" % self.name)
    
    def __call__(self, data, silent=False, **kwargs):
        d = {k: getattr(random, k) for k in ["choice", "randbytes", "randint", "randrange", "randstr"]}
        d.update(_EVAL_NAMESPACE)
        d.update(data)
        kwargs.update(self.parameters)
        try:
            func = eval2(self.result, d, {}, whitelist_nodes=WL_NODES + WL_EXTRA_NODES)
            if len(kwargs) == 0:
                return e
        except Exception as e:
            if not silent:
                self.parent.logger.warning("Bad expression: %s" % self.result)
                self.parent.logger.error(str(e))
                self.parent.logger.debug("Variables:\n- %s" % \
                                         "\n- ".join("%s(%s)=%s" % (k, type(v).__name__, v) for k, v in d.items()))
            raise
        try:
            return func(**kwargs)
        except Exception as e:
            if not silent:
                self.parent.logger.warning("Bad function: %s" % self.result)
                self.parent.logger.error(str(e))
            raise


class MetaBase(type):
    """ This metaclass allows to iterate names over the class-bound registry of the underlying abstraction.
         It also adds a class-level 'source' attribute that can be used to reset the registry. """
    def __iter__(self):
        self(None)  # trigger registry's lazy initialization
        temp = []
        for base in self.registry.values():
            for name in base.keys():
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
        p = Path(str(path or config[self.__name__.lower()]), expand=True)
        if hasattr(self, "_source") and self._source == p:
            return
        self._source = p
        self.registry = None  # reset the registry


def backup(f):
    """ Simple method decorator for making a backup of the dataset. """
    def _wrapper(s, *a, **kw):
        if config['keep_backups']:
            s.backup = s
        return f(s, *a, **kw)
    return _wrapper


def make_registry(cls):
    """ Make class' registry of child classes and fill the __all__ list in. """
    def _setattr(i, d):
        for k, v in d.items():
            if k == "status":
                k = "_" + k
            setattr(i, k, v)
    # open the .conf file associated to cls (i.e. Detector, Packer, ...)
    cls.registry, glob = [], inspect.getparentframe().f_back.f_globals
    with Path(config[cls.__name__.lower() + "s"]).open() as f:
        items = yaml.load(f, Loader=yaml.Loader)
    # start parsing items of cls
    _cache, defaults = {}, items.pop('defaults', {})
    for item, data in items.items():
        for k, v in defaults.items():
            if k in ["base", "install", "status", "steps", "variants"]:
                raise ValueError("parameter '%s' cannot have a default value" % k)
            data.setdefault(k, v)
        # ensure the related item is available in module's globals()
        #  NB: the item may already be in globals in some cases like pbox.items.packer.Ezuri
        if item not in glob:
            d = dict(cls.__dict__)
            del d['registry']
            glob[item] = type(item, (cls, ), d)
        i = glob[item]
        i._instantiable = True
        # before setting attributes from the YAML parameters, check for 'base' ; this allows to copy all attributes from
        #  an entry originating from another item class (i.e. copying from Packer's equivalent to Unpacker ; e.g. UPX)
        base = data.get('base')  # i.e. detector|packer|unpacker ; DO NOT pop as 'base' is also used for algorithms
        if isinstance(base, str):
            m = re.match(r"(?i)(detector|packer|unpacker)(?:\[(.*?)\])?$", str(base))
            if m:
                data.pop('base')
                base, bcls = m.groups()
                base, bcls = base.capitalize(), bcls or item
                if base == cls.__name__ and bcls in [None, item]:
                    raise ValueError("%s cannot point to itself" % item)
                if base not in _cache.keys():
                    with Path(config[base.lower() + "s"]).open() as f:
                        _cache[base] = yaml.load(f, Loader=yaml.Loader)
                for k, v in _cache[base].get(bcls, {}).items():
                    # do not process these keys as they shall be different from an item class to another anyway
                    if k in ["steps", "status"]:
                        continue
                    setattr(i, k, v)
            else:
                raise ValueError("'base' set to '%s' of %s discarded (bad format)" % (base, item))
        # check for eventual variants ; the goal is to copy the current item class and to adapt the fields from the
        #  variants to the new classes (note that on the contrary of base, a variant inherits the 'status' parameter)
        variants, vilist = data.pop('variants', {}), []
        for vitem in variants.keys():
            d = dict(cls.__dict__)
            del d['registry']
            vi = glob[vitem] = type(vitem, (cls, ), d)
            vi._instantiable = True
            vi.parent = item
            vilist.append(vi)
        # now set attributes from YAML parameters
        for it in [i] + vilist:
            _setattr(it, data)
        glob['__all__'].append(item)
        cls.registry.append(i())
        # overwrite parameters specific to variants
        for vitem, vdata in variants.items():
            vi = glob[vitem]
            _setattr(vi, vdata)
            glob['__all__'].append(vitem)
            cls.registry.append(vi())


def purge_items(cls, name):
    """ Purge all items designated by 'name' for the given class 'cls'. """
    purged = False
    if name == "all":
        for obj in cls.iteritems(True):
            obj.purge()
            purged = True
    elif "*" in name:
        name = r"^%s$" % name.replace("*", "(.*)")
        for obj in cls.iteritems(False):
            if re.search(name, obj.stem):
                cls.open(obj).purge()
                purged = True
    else:
        cls.open(name).purge()
        purged = True
    return purged


def select_features(dataset, feature=None):
    """ Handle features selection based on a simple wildcard. """
    if not hasattr(dataset, "_features"):
        dataset._compute_all_features()
    if feature is None or feature == []:
        feature = list(dataset._features.keys())
    # data preparation
    if not isinstance(feature, (tuple, list)):
        feature = [feature]
    nfeature = []
    for pattern in feature:
        # handle wildcard for bulk-selecting features
        if "*" in pattern:
            regex = re.compile(pattern.replace("*", ".*"))
            for f in dataset._features.keys():
                if regex.search(f):
                    nfeature.append(f)
        else:
            nfeature.append(pattern)
    return sorted(nfeature)

