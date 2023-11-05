# -*- coding: UTF-8 -*-
import builtins
from tinyscript import inspect, logging, random, re
from tinyscript.helpers import is_file, is_folder, set_exception, Path, TempPath
from tinyscript.helpers.expressions import WL_NODES


lazy_load_module("yaml")
set_exception("NotInstantiable", "TypeError")


__all__ = ["dict2", "load_yaml_config", "select_features", "Item", "MetaBase", "MetaItem", "TEST_FILES"]

_EVAL_NAMESPACE = {k: getattr(builtins, k) for k in ["abs", "divmod", "float", "hash", "hex", "id", "int", "len",
                                                     "list", "max", "min", "next", "oct", "ord", "pow", "range",
                                                     "range2", "round", "set", "str", "sum", "tuple", "type"]}
TEST_FILES = {
    'ELF32': [
        "/usr/bin/perl",
        "/usr/lib/wine/wine",
        "/usr/lib/wine/wineserver32",
        "/usr/libx32/crti.o",
        "/usr/libx32/libpcprofile.so",
    ],
    'ELF64': [
        "/bin/cat",
        "/bin/ls",
        "/bin/mandb",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/git-core/git",
        "/usr/lib/x86_64-linux-gnu/crti.o",
        "/usr/lib/x86_64-linux-gnu/libpcprofile.so",
        "/usr/lib/ld-linux.so.2",
    ],
    'MSDOS': [
        "~/.wine32/drive_c/windows/rundll.exe",
        "~/.wine32/drive_c/windows/system32/gdi.exe",
        "~/.wine32/drive_c/windows/system32/user.exe",
        "~/.wine32/drive_c/windows/system32/mouse.drv",
        "~/.wine32/drive_c/windows/system32/winaspi.dll",
    ],
    'PE32': [
        "~/.wine32/drive_c/windows/winhlp32.exe",
        "~/.wine32/drive_c/windows/system32/plugplay.exe",
        "~/.wine32/drive_c/windows/system32/winemine.exe",
        "~/.wine32/drive_c/windows/twain_32.dll",
        "~/.wine32/drive_c/windows/twain_32/sane.ds",
        "~/.wine32/drive_c/windows/system32/msscript.ocx",
        "~/.wine32/drive_c/windows/system32/msgsm32.acm",
    ],
    'PE64': [
        "~/.wine64/drive_c/windows/hh.exe",
        "~/.wine64/drive_c/windows/system32/spoolsv.exe",
        "~/.wine64/drive_c/windows/system32/dmscript.dll",
        "~/.wine64/drive_c/windows/twain_64/gphoto2.ds",
        "~/.wine64/drive_c/windows/system32/msscript.ocx",
        "~/.wine64/drive_c/windows/system32/msadp32.acm",
    ],
}
UNDEF_RESULT = "undefined"
WL_EXTRA_NODES = ("arg", "arguments", "keyword", "lambda")


_fmt_name = lambda x: (x or "").lower().replace("_", "-")


class dict2(dict):
    """ Simple extension of dict for defining callable items. """
    def __init__(self, idict, **kwargs):
        self.setdefault("name", UNDEF_RESULT)
        self.setdefault("description", "")
        self.setdefault("result", self['name'])
        for f, v in getattr(self.__class__, "_fields", {}).items():
            self.setdefault(f, v)
        logger = idict.pop('logger', kwargs.pop('logger', null_logger))
        super().__init__(idict, **kwargs)
        self.__dict__ = self
        dict2._logger = logger
        if self.result == UNDEF_RESULT:
            raise ValueError("%s: 'result' shall be defined" % self.name)
    
    def __call__(self, data, silent=False, **kwargs):
        d = {k: getattr(random, k) for k in ["choice", "randbytes", "randint", "randrange", "randstr"]}
        d['apply'] = _apply
        d.update(_EVAL_NAMESPACE)
        d.update(data)
        kwargs.update(getattr(self, "parameters", {}))
        # execute an expression from self.result (can be a single expression or a list of expressions to be chained)
        def _exec(expr):
            try:
                r = eval2(expr, d, {}, whitelist_nodes=WL_NODES + WL_EXTRA_NODES)
                if len(kwargs) == 0:  # means no parameter provided
                    return r
            except NameError as e:
                if not silent:
                    dict2._logger.debug("'%s' is either not computed yet or mistaken" % str(e).split("'")[1])
                raise
            except Exception as e:
                if not silent:
                    dict2._logger.warning("Bad expression: %s" % result)
                    dict2._logger.error(str(e))
                    dict2._logger.debug("Variables:\n- %s" % \
                                      "\n- ".join("%s(%s)=%s" % (k, type(v).__name__, v) for k, v in d.items()))
                return
            try:
                return r(**kwargs)
            except Exception as e:
                if not silent:
                    dict2._logger.warning("Bad function: %s" % result)
                    dict2._logger.error(str(e))
        # now execute expression(s) ; support for multiple expressions must be explicitely enabled for the class
        if getattr(self.__class__, "_multi_expr", False) and isinstance(self.result, (list, tuple)):
            raise ValueError("List of expressions is not supported for the result of %s" % self.__class__.__name__)
        retv = [_exec(result) for result in (self.result if isinstance(self.result, (list, tuple)) else [self.result])]
        return retv[0] if len(retv) == 1 else tuple(retv)


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
    def logger(self):
        if not hasattr(self, "_logger"):
            n = self.__name__.lower()
            self._logger = logging.getLogger(n)
            logging.setLogger(n)
        return self._logger
    
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
    
    def test(self, files=None, keep=False, **kw):
        """ Tests on some executable files. """
        from tinyscript.helpers import execute_and_log as run
        from .formats import expand_formats
        from ..core.executable import Executable
        d = TempPath(prefix="%s-tests-" % self.__name__.lower(), length=8)
        self.__disp = []
        self()  # force registry computation
        for fmt in expand_formats("All"):
            if fmt not in self.registry.keys():
                self.logger.warning("no %s defined yet for '%s'" % (self.__name__.lower().rstrip("s"), fmt))
                continue
            l = [f for f in files if Executable(f).format in self._formats_exp] if files else TEST_FILES.get(fmt, [])
            if len(l) == 0:
                continue
            self.logger.info(fmt)
            for exe in l:
                exe = Executable(exe, expand=True)
                tmp = d.joinpath(exe.filename)
                self.logger.debug(exe.filetype)
                run("cp %s %s" % (exe, tmp))
                n = tmp.filename
                try:
                    self(Executable(tmp))
                    self.logger.success(n)
                except Exception as e:
                    if isinstance(e, KeyError) and exe.format is None:
                        self.logger.error("unknown format (%s)" % exe.filetype)
                        continue
                    self.logger.failure(n)
                    self.logger.exception(e)
        if not keep:
            self.logger.debug("rm -f %s" % str(d))
            d.remove()


def _apply(f):
    """ Simple decorator for applying an operation to the result of the decorated function. """
    def _wrapper(op):
        def _subwrapper(*a, **kw):
            return op(f(*a, **kw))
        return _subwrapper
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
        def logger(self):
            if not hasattr(self, "_logger"):
                n = self.__name__.lower()
                self._logger = logging.getLogger(n)
                logging.setLogger(n)
            return self._logger
        
        @property
        def names(self):
            return sorted(x.name for x in self.registry)
        
        @property
        def source(self):
            if not hasattr(self, "_source"):
                self.source = None
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
            if not p.exists():
                self.logger.warning("'%s' does not exist ; set back to default" % p)
                p, func = config.DEFAULTS['definitions']['%ss' % self.__name__.lower()]
                p = func(p)
            # start parsing items of the target class
            _cache = {}
            for item, data in load_yaml_config(p, ("base", "install", "steps", "variants")):
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
                            _cache[base] = dict(load_yaml_config(base.lower() + "s"))
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
            return self.__class__.logger
        
        @property
        def source(self):
            if self._instantiable and not isinstance(self._source, str):
                self.__class__._source = ""
            return self.__class__._source
    return Item
lazy_load_object("Item", _init_item)


def load_yaml_config(cfg, no_defaults=(), parse_defaults=True):
    """ Load a YAML configuration, either as a single file or a folder with YAML files (in this case, loading
         defaults.yml in priority to get the defaults first). """
    def _set(config):
        if parse_defaults:
            defaults = config.pop('defaults', {})
            for params in [v for v in config.values()]:
                for default, value in defaults.items():
                    if default in no_defaults:
                        raise ValueError("default value for parameter '%s' is not allowed" % default)
                    if isinstance(value, dict):
                        # example advanced defaults configuration:
                        #   defaults:
                        #     keep:
                        #       match:
                        #         - ^entropy*   (do not keep entropy-based features)
                        #         - ^is_*       (do not keep boolean features)
                        #       value: false
                        value.pop('comment', None)
                        if set(value.keys()) != {'match', 'value'}:
                            raise ValueError("Bad configuration for default '%s' ; should be a dictionary with 'value'"
                                             " and 'match' (format: list) as its keys" % default)
                        v = value['value']
                        for pattern in value['match']:
                            for name2 in config.keys():
                                # special case: boolean value ; in this case, set value for matching names and set
                                #                                non-matching names to its opposite
                                if isinstance(v, bool):
                                    # in the advanced example here above, entropy-based and boolean features will not be
                                    #  kept but all others will be
                                    config[name2].setdefault(default, v if re.search(pattern, name2) else not v)
                                    # this means that, if we want to keep additional features, we can still force
                                    #  keep=true per feature declaration
                                elif re.search(pattern, name2):
                                    config[name2].setdefault(default, v)
                    else:
                        # example normal defaults configuration:
                        #   defaults:
                        #     keep:   false
                        #     source: <unknown>
                        params.setdefault(default, value)
        return config
    # get the list of configs ; may be:
    #  - single YAML file
    #  - folder with YAML files
    p = cfg if isinstance(cfg, Path) else Path(config[cfg])
    configs = list(p.listdir(lambda x: x.extension == ".yml")) if p.is_dir() else [p]
    # separate defaults.yml from the list
    defaults = list(filter(lambda x: x.stem == "defaults", configs))
    configs = list(filter(lambda x: x.stem != "defaults", configs))
    d = {}
    if len(defaults) > 0:
        with defaults[0].open() as f:
            d = yaml.load(f, Loader=yaml.Loader)
    # now parse YAML configurations, setting local defaults from child configs
    for c in configs:
        with c.open() as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        d.update(_set(cfg or {}))
    # collect properties that are applicable for all the other features
    for name, params in _set(d).items():
        yield name, params


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

