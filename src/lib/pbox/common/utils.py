# -*- coding: UTF-8 -*-
import pandas as pd
import re
import yaml
from functools import wraps
from time import perf_counter, time
from tinyscript import inspect, logging, subprocess
from tinyscript.helpers import is_file, is_folder, Path
try:  # from Python3.9
    import mdv3 as mdv
except ImportError:
    import mdv

from .config import config


__all__ = ["aggregate_formats", "backup", "benchmark", "class_or_instance_method", "collapse_formats",
           "edit_file", "expand_formats", "file_or_folder_or_dataset", "highlight_best", "make_registry", "mdv",
           "metrics", "shorten_str", "FORMATS", "PERF_HEADERS"]


FORMATS = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     [".NET", "PE32", "PE64"],
}
PERF_HEADERS = {
    'Dataset':         lambda x: x,
    'Accuracy':        lambda x: "-" if x == "-" else "%.2f%%" % (x * 100),
    'Precision':       lambda x: "-" if x == "-" else "%.2f%%" % (x * 100),
    'Recall':          lambda x: "-" if x == "-" else "%.2f%%" % (x * 100),
    'F-Measure':       lambda x: "-" if x == "-" else "%.2f%%" % (x * 100),
    'MCC':             lambda x: "-" if x == "-" else "%.2f%%" % (x * 100),
    'AUC':             lambda x: "-" if x == "-" else "%.2f%%" % (x * 100),
    'Processing Time': lambda x: "%.3fms" % (x * 1000),
}


bold = lambda text: "\033[1m{}\033[0m".format(text)


def aggregate_formats(*formats, **kw):
    """ Aggregate the given input formats. """
    l = []
    for c in formats:
        if isinstance(c, (list, tuple)):
            l.extend(expand_formats(*c))
        else:
            l.append(c)
    return collapse_formats(*set(l)) if kw.get('collapse', False) else list(set(l))


def backup(f):
    """ Simple method decorator for making a backup of the dataset. """
    def _wrapper(s, *a, **kw):
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


def edit_file(path, csv_sep=";", **kw):
    """" Edit a target file with visidata. """
    cmd = "vd %s --csv-delimiter \"%s\"" % (path, csv_sep)
    l = kw.pop('logger', None)
    if l:
        l.debug(cmd)
    subprocess.call(cmd, stderr=subprocess.PIPE, shell=True, **kw)


def expand_formats(*formats, **kw):
    """ 2-depth dictionary-based expansion function for resolving a list of executable formats. """
    selected = []
    for c in formats:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
        for sc in FORMATS.get(c, [c]):   # depth 2: e.g. ELF => ELF32,ELF64
            if kw.get('once', False):
                selected.append(sc)
            else:
                for ssc in FORMATS.get(sc, [sc]):
                    if ssc not in selected:
                        selected.append(ssc)
    return selected


def file_or_folder_or_dataset(method):
    """ This decorator allows to handle, as the first positional argument of an instance method, either an executable,
         a folder with executables or the executable files from a Dataset. """
    @wraps(method)
    def _wrapper(self, *args, **kwargs):
        # collect executables and folders from args
        n, e, l = -1, [], {}
        # exe list extension function
        def _extend_e(i):
            nonlocal n, e, l
            # append the (Fileless)Dataset instance itself
            if getattr(i, "is_valid", lambda: False)():
                for exe in i._iter_with_features(kwargs.get('feature'), kwargs.get('pattern')):
                    e.append(exe)
            # single executable
            elif is_file(i) and i not in e:
                i = Path(i)
                i.dataset = None
                e.append(i)
            # normal folder or FilelessDataset's path or Dataset's files path
            elif is_folder(i):
                for f in Path(i).listdir():
                    f.dataset = None
                    if str(f) not in e:
                        e.append(f)
            else:
                i = config['datasets'].joinpath(i)
                # check if it has the structure of a dataset
                if i.joinpath("files").is_dir() and not i.joinpath("features.json").exists() and \
                   all(i.joinpath(f).is_file() for f in ["data.csv", "metadata.json"]) or \
                   not i.joinpath("files").exists() and \
                   all(i.joinpath(f).is_file() for f in ["data.csv", "features.json", "metadata.json"]):
                    data = pd.read_csv(str(i.joinpath("data.csv")), sep=";")
                    l = {e.hash: e.label for e in data.itertuples()}
                    dataset = i.basename
                    # if so, move to the dataset's "files" folder
                    if not i.joinpath("files").exists():
                        for h in data.hash.values:
                            p = i.joinpath(h)
                            if p not in e:
                                e.append(p)
                        return True
                    else:
                        i = i.joinpath("files")
                if is_folder(i):
                    for f in i.listdir():
                        f.dataset = dataset
                        if str(f) not in e:
                            e.append(f)
                else:
                    return False
            return True
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
        kwargs['silent'] = kwargs.get('silent', False)
        for exe in e:
            kwargs['dslen'] = len(e)
            # this is useful for a decorated method that handles the difference between the computed and actual labels
            lbl = l.get(Path(exe).stem, -1)
            kwargs['label'] = [lbl, None][str(lbl) == "nan"]
            yield method(self, exe, *args, **kwargs)
            kwargs['silent'] = True
    return _wrapper


def highlight_best(data, headers=None, exclude_cols=[0, -1], formats=None):
    """ Highlight the highest values in the given table. """
    if len(data[0]) != len(headers):
        raise ValueError("headers and row lengths mismatch")
    ndata, exc_cols = [], [x % len(headers) for x in exclude_cols]
    maxs = [None if i in exc_cols else 0 for i, _ in enumerate(headers)]
    fl = lambda f: -1. if f == "-" else float(f)
    # search for best values
    for d in data:
        for i, v in enumerate(d):
            if maxs[i] is None:
                continue
            maxs[i] = max(maxs[i], fl(v))
    # reformat the table, setting bold text for best values
    for d in data:
        ndata.append([bold((formats or {}).get(k, lambda x: x)(v)) if maxs[i] and fl(v) == maxs[i] else \
                     (formats or {}).get(k, lambda x: x)(v) for i, (k, v) in enumerate(zip(headers, d))])
    return ndata


def make_registry(cls):
    """ Make class' registry of child classes and fill the __all__ list in. """
    def _setattr(i, d):
        for k, v in d.items():
            if k == "status":
                k = "_" + k
            setattr(i, k, v)
    # open the .conf file associated to cls (i.e. Detector, Packer, ...)
    cls.registry, glob = [], inspect.getparentframe().f_back.f_globals
    with Path("/opt/%ss.yml" % cls.__name__.lower()).open() as f:
        items = yaml.load(f, Loader=yaml.Loader)
    # start parsing items of cls
    _cache = {}
    for item, data in items.items():
        # ensure the related item is available in module's globals()
        #  NB: the item may already be in globals in some cases like pbox.items.packer.Ezuri
        if item not in glob:
            glob[item] = type(item, (cls, ), dict(cls.__dict__))
        i = glob[item]
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
                    with Path("/opt/%ss.yml" % base.lower()).open() as f:
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
            vi = glob[vitem] = type(vitem, (cls, ), dict(cls.__dict__))
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


def metrics(tn=0, fp=0, fn=0, tp=0):
    """ Compute some metrics related to false/true positives/negatives. """
    accuracy  = float(tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else -1
    precision = float(tp) / (tp + fp) if tp + fp > 0 else -1
    recall    = float(tp) / (tp + fn) if tp + fn > 0 else -1                                      # or also sensitivity
    f_measure = 2. * precision * recall / (precision + recall) if precision + recall > 0 else -1  # or F(1)-score
    return accuracy, precision, recall, f_measure


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


# based on: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, ins, typ):
        return (super().__get__ if ins is None else self.__func__.__get__)(ins, typ)

