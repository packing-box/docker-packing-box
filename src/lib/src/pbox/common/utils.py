# -*- coding: UTF-8 -*-
import builtins
import pandas as pd
import re
import yaml
from contextlib import contextmanager
from functools import wraps
from time import perf_counter, time
from tinyscript import ast, inspect, logging, os, random, subprocess
from tinyscript.helpers import is_file, is_folder, Path, TempPath
from tinyscript.helpers.expressions import WL_NODES
from tqdm import tqdm

from .config import *
from .executable import Executable


__all__ = ["aggregate_formats", "backup", "benchmark", "bin_label", "class_or_instance_method", "collapse_formats",
           "data_to_temp_file", "dict2", "edit_file", "expand_formats", "expand_parameters",
           "file_or_folder_or_dataset", "filter_data", "get_counts", "is_exe", "make_registry", "shorten_str",
           "strip_version", "tqdm", "ExeFormatDict", "COLORMAP", "FORMATS"]

_EVAL_NAMESPACE = {k: getattr(builtins, k) for k in ["abs", "divmod", "float", "hash", "hex", "id", "int", "len",
                                                     "list", "max", "min", "oct", "ord", "pow", "range", "range2",
                                                     "round", "set", "str", "sum", "tuple", "type"]}
WL_EXTRA_NODES = ("arg", "arguments", "keyword", "lambda")

COLORMAP = {
    'red':        (255, 0,   0),
    'lightCoral': (240, 128, 128),
    'purple':     (128, 0,   128),
    'peru':       (205, 133, 63),
    'salmon':     (250, 128, 114),
    'rosyBrown':  (188, 143, 143),
    'sandyBrown': (244, 164, 96),
    'sienna':     (160, 82,  45),
    'plum':       (221, 160, 221),
    'pink':       (255, 192, 203),
    'tan':        (210, 180, 140),
    'tomato':     (255, 99,  71),
    'violet':     (238, 130, 238),
    'magenta':    (255, 0,   255),
    'fireBrick':  (178, 34,  34),
    'indigo':     (75,  0,   130),
}

FORMATS = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     [".NET", "PE32", "PE64"],
}


bin_label = lambda l: {NOT_LABELLED.lower(): -1, 'false': 0, NOT_PACKED.lower(): 0, 'true': 1, None: None} \
                      .get(l.lower(), 1)
bold = lambda text: "\033[1m{}\033[0m".format(text)
get_counts = lambda metadata, packed=True: {k: v for k, v in metadata['counts'].items() if k not in \
                                            ([NOT_LABELLED, NOT_PACKED] if packed else [NOT_LABELLED])}
is_exe = lambda e: Executable(e).format is not None


class dict2(dict):
    """ Simple extension of dict for defining callable items. """
    def __init__(self, idict, **kwargs):
        self.setdefault("name", "undefined")
        self.setdefault("description", "")
        self.setdefault("result", None)
        self.setdefault("parameters", {})
        for f, v in getattr(self.__class__, "_fields", {}).items():
            self.setdefault(f, v)
        super(dict2, self).__init__(idict, **kwargs)
        self.__dict__ = self
        if self.result is None:
            raise ValueError("%s: 'result' shall be defined" % self.name)
    
    def __call__(self, data, silent=False, **kwargs):
        d = {k: getattr(random, k) for k in ["choice", "randint", "randrange", "randstr"]}
        d.update(_EVAL_NAMESPACE)
        d.update(data)
        kwargs.update(self.parameters)
        try:
            e = eval2(self.result, d, {}, whitelist_nodes=WL_NODES + WL_EXTRA_NODES)
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
            return e(**kwargs)
        except Exception as e:
            if not silent:
                self.parent.logger.warning("Bad function: %s" % self.result)
                self.parent.logger.error(str(e))
            raise


class ExeFormatDict(dict):
    """ Special dictionary for handling aggregates of sub-dictionaries applying to an executable format, a class of
         formats (depth 1: PE, ELF, Mach-O) or any format (depth 0: All).

    depth 0: All
    depth 1: PE, ELF, Mach-O
    depth 2: PE32, PE64, ELF32, ...
    """
    def __init__(self, *args, **kwargs):
        self.__all = expand_formats("All")
        self.__get = super(ExeFormatDict, self).__getitem__
        d = args[0] if len(args) > 0 else {}
        d.update(kwargs)
        for i in range(3):
            self.setdefault(i, {})
        for k, v in d.items():
            self[k] = v

    def __delitem__(self, name):
        depth = 0 if name == "All" else 1 if name in FORMATS.keys() else 2 if name in self.__all else -1
        if depth == -1:
            raise ValueError("Unhandled key '%s'" % name)
        del self.__get(depth)[name]

    def __getitem__(self, name):
        if name not in self.__all:
            raise ValueError("Bad executable format")
        r = self.__get(0)['All']
        fcls = [k for k in self.__get(1).keys() if name in FORMATS[k]]
        if len(fcls) > 0:
            r.update(self.__get(1)[fcls[0]])
        r.update(self.__get(2).get(name, {}))
        return r

    def __setitem__(self, name, value):
        update = False
        if isinstance(name, (list, tuple)) and len(name) == 2:
            name, update = name
        depth = 0 if name == "All" else 1 if name in FORMATS.keys() else 2 if name in self.__all else -1
        if depth == -1:
            raise ValueError("Unhandled key '%s'" % name)
        if update:
            self.__get(depth)[name].update(value)
        else:
            self.__get(depth)[name] = value


def aggregate_formats(*formats, **kw):
    """ Aggregate the given input formats. """
    l = []
    for f in formats:
        if isinstance(f, (list, tuple)):
            l.extend(expand_formats(*f))
        else:
            l.append(f)
    return collapse_formats(*set(l)) if kw.get('collapse', False) else list(set(l))


def backup(f):
    """ Simple method decorator for making a backup of the dataset. """
    def _wrapper(s, *a, **kw):
        if config['keep_backups']:
            s.backup = s
        return f(s, *a, **kw)
    return _wrapper


def benchmark(f):
    """ Decorator for benchmarking function executions. """
    def _wrapper(*args, **kwargs):
        t = perf_counter if kwargs.pop("perf", True) else time
        start = t()
        r = f(*args, **kwargs)
        dt = t() - start
        return r, dt
    return _wrapper


def collapse_formats(*formats, **kw):
    """ 2-depth dictionary-based collapsing function for getting a short list of executable formats. """
    # also support list input argument
    if len(formats) == 1 and isinstance(formats[0], (tuple, list)):
        formats = formats[0]
    selected = [x for x in formats]
    groups = [k for k in FORMATS.keys() if k != "All"]
    for c in groups:
        # if a complete group of formats (PE, ELF, Mach-O) is included, only keep the entire group
        if all(x in selected for x in FORMATS[c]):
            for x in FORMATS[c]:
                selected.remove(x)
            selected.append(c)
    # ensure children of complete groups are removed
    for c in selected[:]:
        if c in groups:
            for sc in selected:
                if sc in FORMATS[c]:
                    selected.remove(sc)
    # if everything in the special group 'All' is included, simply select only 'All'
    if all(x in selected for x in FORMATS['All']):
        selected = ["All"]
    return list(set(selected))


@contextmanager
def data_to_temp_file(data, prefix="temp"):
    """ Save the given pandas.DataFrame to a temporary file. """
    p = TempPath(prefix=prefix, length=8)
    f = p.tempfile("data.csv")
    data.to_csv(str(f), sep=";", index=False, header=True)
    yield f
    p.remove()


def edit_file(path, csv_sep=";", text=False, **kw):
    """" Edit a target file with visidata. """
    cmd = "%s %s" % (os.getenv('EDITOR'), path) if text else "vd %s --csv-delimiter \"%s\"" % (path, csv_sep)
    l = kw.pop('logger', None)
    if l:
        l.debug(cmd)
    subprocess.call(cmd, stderr=subprocess.PIPE, shell=True, **kw)


def expand_formats(*formats, **kw):
    """ 2-depth dictionary-based expansion function for resolving a list of executable formats. """
    selected = []
    for f in formats:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
        for sf in FORMATS.get(f, [f]):   # depth 2: e.g. ELF => ELF32,ELF64
            if kw.get('once', False):
                selected.append(sf)
            else:
                for ssc in FORMATS.get(sf, [sf]):
                    if ssc not in selected:
                        selected.append(ssc)
    return selected


def expand_parameters(*strings, **kw):
    """ This simple helper expands a [sep]-separated string of keyword-arguments defined with [key]=[value]. """
    sep = kw.get('sep', ",")
    d = {}
    for s in strings:
        for p in s.split(sep):
            k, v = p.split("=", 1)
            try:
                v = ast.literal_eval(v)
            except ValueError:
                pass
            d[k] = v
    return d


def file_or_folder_or_dataset(method):
    """ This decorator allows to handle, as the first positional argument of an instance method, either an executable,
         a folder with executables or the executable files from a Dataset. """
    @wraps(method)
    def _wrapper(self, *args, **kwargs):
        kwargs['silent'] = kwargs.get('silent', False)
        # collect executables and folders from args
        n, e, l = -1, [], {}
        # exe list extension function
        def _extend_e(i):
            nonlocal n, e, l
            # append the (Fileless)Dataset instance itself
            if not isinstance(i, Executable) and getattr(i, "is_valid", lambda: False)():
                if not kwargs['silent']:
                    self.logger.debug("input is a (Fileless)Dataset structure")
                for exe in i:
                    e.append(exe)
                return True
            # single executable
            elif is_file(i):
                if not kwargs['silent']:
                    self.logger.debug("input is a single executable")
                if i not in e:
                    i = Path(i)
                    i.dataset = None
                    e.append(i)
                return True
            # normal folder or FilelessDataset's path or Dataset's files path
            elif is_folder(i):
                if not kwargs['silent']:
                    self.logger.debug("input is a folder of executables")
                for f in Path(i).walk(filter_func=lambda p: p.is_file()):
                    f.dataset = None
                    if str(f) not in e:
                        e.append(f)
                return True
            else:
                i = config['datasets'].joinpath(i)
                # check if it has the structure of a dataset
                if all(i.joinpath(f).is_file() for f in ["data.csv", "metadata.json"]):
                    if i.joinpath("files").is_dir() and not i.joinpath("features.json").exists():
                       
                        if not kwargs['silent']:
                            self.logger.debug("input is Dataset from %s" % config['datasets'])
                        data = pd.read_csv(str(i.joinpath("data.csv")), sep=";")
                        l = {exe.hash: exe.label for exe in data.itertuples()}
                        dataset = i.basename
                        for f in i.joinpath("files").listdir():
                            f.dataset = dataset
                            if str(f) not in e:
                                e.append(f)
                        return True
                    # otherwise, it is a FilelessDataset and it won't work as this decorator requires samples
                    self.logger.warning("FilelessDataset is not supported as it does not hold samples to iterate")
            return False
        # use the extension function to parse:
        # - positional arguments up to the last valid file/folder
        # - then the 'file' keyword-argument
        for n, a in enumerate(args):
            # if not a valid file, folder or dataset, stop as it is another type of argument
            if not _extend_e(a):
                break
        args = tuple(args[n+1:])
        for a in kwargs.pop('file', []):
            _extend_e(a)
        # then handle the list
        i, kwargs['silent'] = -1, kwargs.get('silent', False)
        for i, exe in enumerate(e):
            exe = Executable(exe)
            if exe.format is None:  # format is not in the executable SIGNATURES of pbox.common.executable
                self.logger.debug("'%s' is not a valid executable" % exe)
                continue
            kwargs['dslen'] = len(e)
            # this is useful for a decorated method that handles the difference between the computed and actual labels
            kwargs['label'] = l.get(Path(exe).stem, NOT_LABELLED)
            try:
                yield method(self, exe, *args, **kwargs)
            except ValueError as err:
                self.logger.exception(err)
            kwargs['silent'] = True
        if i == -1:
            self.logger.error("No (valid) executable selected")
    return _wrapper


def filter_data(df, query=None, **kw):
    """ Fitler an input Pandas DataFrame based on a given query. """
    i, l = -1, kw.get('logger', null_logger)
    if query is None or query.lower() == "all":
        return df
    try:
        r = df.query(query)
        if len(r) == 0:
            l.warning("No data selected")
        return r
    except (AttributeError, KeyError) as e:
        l.error("Invalid query syntax ; %s" % e)
    except SyntaxError:
        l.error("Invalid query syntax ; please checkout Pandas' documentation for more information")
    except pd.errors.UndefinedVariableError as e:
        l.error(e)
        l.info("Possible values:\n%s" % "".join("- %s\n" % n for n in df.columns))


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


def shorten_str(string, l=80):
    """ Shorten a string, possibly represented as a comma-separated list. """
    i = 0
    if len(string) <= l:
        return string
    s = ",".join(string.split(",")[:-1])
    if len(s) == 0:
        return string[:l-3] + "..."
    while 1:
        t = s.split(",")
        if len(t) > 1:
            s = ",".join(t[:-1])
            if len(s) < l-3:
                return s + "..."
        else:
            return s[:l-3] + "..."
    return s + "..."


def strip_version(name):
    """ Simple helper to strip the version number from a name (e.g. a packer). """
    # name pattern is assumed to be hyphen-separated tokens ; the last one is checked for a version pattern
    if re.match(r"^(\d+\.)*(\d+)([\._]?(\d+|[a-zA-Z]\d*|alpha|beta))?$", name.split("-")[-1]):
        return "-".join(name.split("-")[:-1])
    return name


# based on: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, ins, typ):
        return (super().__get__ if ins is None else self.__func__.__get__)(ins, typ)

