# -*- coding: UTF-8 -*-
import pandas as pd
import yaml
from functools import wraps
from time import perf_counter, time
from tinyscript import inspect
from tinyscript.helpers import is_executable, is_file, is_folder, Path
try:  # from Python3.9
    import mdv3 as mdv
except ImportError:
    import mdv


__all__ = ["backup", "benchmark", "class_or_instance_method", "collapse_categories", "expand_categories",
           "file_or_folder_or_dataset", "highlight_best", "make_registry", "mdv", "shorten_str", "CATEGORIES"]


CATEGORIES = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     [".NET", "PE32", "PE64"],
}


bold = lambda text: "\033[1m{}\033[0m".format(text)


def backup(f):
    """ Simple method decorator for making a backup of the dataset. """
    def _wrapper(s, *a, **kw):
        s.backup = s
        return f(s, *a, **kw)
    return _wrapper


def benchmark(f):
    """ Decorator for benchmarking function executions. """
    def _wrapper(*args, **kwargs):
        logger, perf = kwargs.get("logger"), kwargs.pop("perf", True)
        t = perf_counter if perf else time
        start = t()
        r = f(*args, **kwargs)
        dt = t() - start
        return r, dt
    return _wrapper


def collapse_categories(*categories, **kw):
    """ 2-depth dictionary-based collapsing function for getting a short list of executable categories. """
    if len(categories) == 1 and isinstance(categories[0], (tuple, list)):
        categories = categories[0]
    selected = [x for x in categories]
    for c in [k for k in CATEGORIES.keys() if k != "All"]:
        if all(x in selected for x in CATEGORIES[c]):
            for x in CATEGORIES[c]:
                selected.remove(x)
            selected.append(c)
    if all(x in selected for x in CATEGORIES['All']):
        selected = ["All"]
    return list(set(selected))


def expand_categories(*categories, **kw):
    """ 2-depth dictionary-based expansion function for resolving a list of executable categories. """
    selected = []
    for c in categories:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
        for sc in CATEGORIES.get(c, [c]):   # depth 2: e.g. ELF => ELF32,ELF64
            if kw.get('once', False):
                selected.append(sc)
            else:
                for ssc in CATEGORIES.get(sc, [sc]):
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
            # append the (Fileless)Dataset instance itself
            if getattr(i, "is_valid", lambda: False)():
                for exe in i._iter_with_features(kwargs.get('feature'), kwargs.get('pattern')):
                    e.append(exe)
            # single executable
            elif is_file(i) and is_executable(i) and i not in e:
                i = Path(i)
                i.dataset = None
                e.append(i)
            # normal folder or FilelessDataset's path or Dataset's files path
            elif is_folder(i):
                i, dataset = Path(i), None
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
                for f in i.listdir(is_executable):
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
        kwargs['silent'] = False
        for exe in e:
            # this is useful for a decorated method that handles the difference between the computed and actual labels
            kwargs['label'] = l.get(Path(exe).stem, -1)
            yield method(self, exe, *args, **kwargs)
            kwargs['silent'] = True
    return _wrapper


def highlight_best(data, headers=None, exclude_cols=[0, -1], formats=None):
    """ Highlight the highest values in the given table. """
    if len(data[0]) != len(headers):
        raise ValueError("headers and row lengths mismatch")
    ndata, exc_cols = [], [x % len(headers) for x in exclude_cols]
    maxs = [None if i in exc_cols else 0 for i, _ in enumerate(headers)]
    # search for best values
    for d in data:
        for i, v in enumerate(d):
            if maxs[i] is None:
                continue
            maxs[i] = max(maxs[i], float(v))
    # reformat the table, setting bold text for best values
    for d in data:
        ndata.append([bold((formats or {}).get(k, lambda x: x)(v)) if maxs[i] and float(v) == maxs[i] else \
                     (formats or {}).get(k, lambda x: x)(v) for i, (k, v) in enumerate(zip(headers, d))])
    return ndata


def make_registry(cls):
    """ Make class' registry of child classes and fill the __all__ list in. """
    cls.registry = []
    glob = inspect.getparentframe().f_back.f_globals
    with Path("/opt/%ss.yml" % cls.__name__.lower()).open() as f:
        items = yaml.load(f, Loader=yaml.Loader)
    for item, data in items.items():
        if item not in glob:
            i = glob[item] = type(item, (cls, ), dict(cls.__dict__))
        else:
            i = glob[item]
        for k, v in data.items():
            if k == "status":
                k = "_" + k
            setattr(i, k, v)
        glob['__all__'].append(item)
        cls.registry.append(i())


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

