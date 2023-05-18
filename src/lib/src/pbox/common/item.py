# -*- coding: UTF-8 -*-
from tinyscript import functools, inspect, logging, re
from tinyscript.helpers import lazy_load_module, lazy_load_object, set_exception, Path
from tinyscript.report import *

from .config import config

lazy_load_module("yaml")


__all__ = ["update_logger", "Item"]


_fmt_name = lambda x: (x or "").lower().replace("_", "-")

set_exception("NotInstantiable", "TypeError")


def update_logger(m):
    """ Method decorator for triggering the setting of the bound logger (see pbox.common.Item.__getattribute__). """
    @functools.wraps(m)
    def _wrapper(self, *a, **kw):
        getattr(self, "logger", None)
        return m(self, *a, **kw)
    return _wrapper


def _init_metaitem():
    class MetaItem(type):
        def __getattribute__(self, name):
            # this masks some attributes for child classes (e.g. Packer.registry can be accessed, but when the registry
            #  of child classes is computed, the child classes, e.g. UPX, won't be able to get UPX.registry)
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__name__, name))
            return super(MetaItem, self).__getattribute__(name)
        
        @property
        def names(self):
            return sorted(x.name for x in self.registry)
        
        @property
        def source(self):
            return self._source
        
        @source.setter
        def source(self, path):
            # case 1: self is a parent class among Analyzer, Detector, ... ;
            #          then 'source' means the source path for loading child classes
            try:
                p = Path(str(path or config['%ss' % self.__name__.lower()]), expand=True)
                if hasattr(self, "_source") and self._source == p:
                    return
                self._source = p
            # case 2: self is a child class of Analyzer, Detector, ... ;
            #          then 'source' is an attribute that comes from the YAML definition
            except KeyError:
                return
            # now make the registry from the given source path
            def _setattr(i, d):
                for k, v in d.items():
                    if k in ["source", "status"]:
                        setattr(i, "_" + k, v)
                    elif hasattr(i, "parent") and k in ["install", "references", "steps"]:
                        nv = []
                        for l in v:
                            if l == "<from-parent>":
                                for l2 in getattr(glob[i.parent], k, []):
                                    nv.append(l2)
                            else:
                                nv.append(l)
                        setattr(i, k, nv)
                    else:
                        setattr(i, k, v)
            # open the .conf file associated to the main class (i.e. Detector, Packer, ...)
            glob = inspect.getparentframe().f_back.f_globals
            # remove the child classes of the former registry from the global scope
            for cls in getattr(self, "registry", []):
                glob.pop(cls.cname, None)
            # reset the registry
            self.registry = []
            if not p.is_file():
                self.logger.warning("'%s' does not exist ; set back to default" % p)
                p, func = config.DEFAULTS['definitions']['%ss' % self.__name__.lower()]
                p = func(p)
            with p.open() as f:
                items = yaml.load(f, Loader=yaml.Loader)
            # start parsing items of the target class
            _cache, defaults = {}, items.pop('defaults', {})
            for item, data in items.items():
                for k, v in defaults.items():
                    if k in ["base", "install", "status", "steps", "variants"]:
                        raise ValueError("parameter '%s' cannot have a default value" % k)
                    data.setdefault(k, v)
                # ensure the related item is available in module's globals()
                #  NB: the item may already be in globals in some cases like pbox.items.packer.Ezuri
                if item not in glob:
                    d = dict(self.__dict__)
                    del d['registry']
                    glob[item] = type(item, (self, ), d)
                i = glob[item]
                i._instantiable = True
                # before setting attributes from the YAML parameters, check for 'base' ; this allows to copy all
                #  attributes from an entry originating from another item class (i.e. copying from Packer's equivalent
                #  to Unpacker ; e.g. UPX)
                base = data.get('base')  # detector|packer|unpacker ; DO NOT pop as 'base' is also used for algorithms
                if isinstance(base, str):
                    m = re.match(r"(?i)(detector|packer|unpacker)(?:\[(.*?)\])?$", str(base))
                    if m:
                        data.pop('base')
                        base, bcls = m.groups()
                        base, bcls = base.capitalize(), bcls or item
                        if base == self.__name__ and bcls in [None, item]:
                            raise ValueError("%s cannot point to itself" % item)
                        if base not in _cache.keys():
                            with Path(config[base.lower() + "s"]).open() as f:
                                _cache[base] = yaml.load(f, Loader=yaml.Loader)
                        for k, v in _cache[base].get(bcls, {}).items():
                            # do not process these keys as they shall be different from an item class to another anyway
                            if k in ["steps", "status"]:
                                continue
                            setattr(i, "_" + k if k == "source" else k, v)
                    else:
                        raise ValueError("'base' set to '%s' of %s discarded (bad format)" % (base, item))
                # check for variants ; the goal is to copy the current item class and to adapt the fields from the
                #  variants to the new classes (note that on the contrary of base, a variant inherits the 'status'
                #  parameter)
                variants, vilist = data.pop('variants', {}), []
                for vitem in variants.keys():
                    d = dict(self.__dict__)
                    del d['registry']
                    vi = glob[vitem] = type(vitem, (self, ), d)
                    vi._instantiable = True
                    vi.parent = item
                    vilist.append(vi)
                # now set attributes from YAML parameters
                for it in [i] + vilist:
                    _setattr(it, data)
                self.registry.append(i())
                # overwrite parameters specific to variants
                for vitem, vdata in variants.items():
                    vi = glob[vitem]
                    _setattr(vi, vdata)
                    self.registry.append(vi())
        
        @property
        def logger(self):
            # important note: ' + " "' allows to fix a name clash with loggers ; e.g. when running the 'detector' tool
            #                  with a single detector, its logger gets the name 'detector', taking precedence on the
            #                  logger of the Detector class => adding " " gives a different string, yet displaying the
            #                  same word.
            n = self.__name__.lower() + " "
            if not hasattr(self, "__logger"):
                self.__logger = logging.getLogger(n)
                logging.setLogger(n)
            return self.__logger
    return MetaItem
lazy_load_object("MetaItem", _init_metaitem)


def _init_item():
    global MetaItem
    MetaItem.__name__  # force the initialization of MetaItem
    class Item(metaclass=MetaItem):
        """ Item abstraction. """
        _instantiable = False
        
        def __init__(self, **kwargs):
            cls = self.__class__
            self.cname = cls.__name__
            self.name = _fmt_name(cls.__name__)
            self.type = cls.__base__.__name__.lower()
        
        def __new__(cls, *args, **kwargs):
            """ Prevents Item from being instantiated. """
            if cls._instantiable:
                return object.__new__(cls, *args, **kwargs)
            raise NotInstantiable("%s cannot be instantiated directly" % cls.__name__)
        
        def __getattribute__(self, name):
            # this masks some attributes for child instances in the same way as for child classes
            if name in ["get", "iteritems", "mro", "registry"] and self._instantiable:
                raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
            return super(Item, self).__getattribute__(name)
        
        def __repr__(self):
            """ Custom string representation for an item. """
            return "<%s %s at 0x%x>" % (self.__class__.__name__, self.type, id(self))
        
        def help(self):
            """ Returns a help message in Markdown format. """
            md = Report()
            if getattr(self, "description", None):
                md.append(Text(self.description))
            if getattr(self, "comment", None):
                md.append(Blockquote("**Note**: " + self.comment))
            if getattr(self, "link", None):
                md.append(Blockquote("**Link**: " + self.link))
            if getattr(self, "references", None):
                md.append(Section("References"), List(*self.references, **{'ordered': True}))
            return md.md()
        
        @classmethod
        def get(cls, item, error=True):
            """ Simple class method for returning the class of an item based on its name (case-insensitive). """
            for i in cls.registry:
                if i.name == (item.name if isinstance(item, Item) else _fmt_name(item)):
                    return i
            if error:
                raise ValueError("'%s' is not defined" % item)
        
        @classmethod
        def iteritems(cls):
            """ Class-level iterator for returning enabled items. """
            for i in cls.registry:
                try:
                    if i.status in i.__class__._enabled:
                        yield i
                except AttributeError:
                    yield i
        
        @property
        def logger(self):
            if not hasattr(self, "__logger"):
                self.__logger = logging.getLogger(self.name)
            logging.setLogger(self.name)
            return self.__logger
        
        @property
        def source(self):
            return self.__class__._source
    return Item
lazy_load_object("Item", _init_item)

