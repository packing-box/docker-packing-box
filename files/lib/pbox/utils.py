# -*- coding: UTF-8 -*-
from functools import wraps
from time import time
from tinyscript.helpers import is_executable, is_file, is_folder, ConfigPath, Path
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
try:  # from Python3.9
    import mdv3 as mdv
except ImportError:
    import mdv


__all__ = ["benchmark", "config", "class_or_instance_method", "collapse_categories", "expand_categories",
           "file_or_folder_or_dataset", "highlight_best", "mdv"]


CATEGORIES = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     [".NET", "PE32", "PE64"],
}


bold = lambda text: "\033[1m{}\033[0m".format(text)


def benchmark(f):
    """ Decorator for benchmarking function executions. """
    def _wrapper(*args, **kwargs):
        logger = kwargs.get("logger")
        info = kwargs.pop("info", None)
        perf = kwargs.pop("perf", True)
        t = perf_counter if perf else time
        start = t()
        r = f(*args, **kwargs)
        dt = t() - start
        message = "{}{}: {} seconds".format(f.__name__, "" if info is None else "[{}]".format(info), dt)
        if logger is None:
            print(message)
        else:
            logger.debug(message)
        return r
    return _wrapper


def collapse_categories(*categories, **kw):
    """ 2-depth dictionary-based collapsing function for getting a short list of executable categories. """
    selected = [x for x in categories]
    for c in [k for k in CATEGORIES.keys() if k != "All"]:
        if all(x in selected for x in CATEGORIES[c]):
            for x in CATEGORIES[c]:
                selected.remove(x)
            selected.append(c)
    if all(x in selected for x in CATEGORIES['All']):
        selected = ["All"]
    return selected


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
            if is_file(i) and is_executable(i) and i not in e:
                e.append(i)
            elif is_folder(i):
                i = Path(i)
                # check if it is a dataset
                if i.joinpath("files").is_dir() and \
                   all(i.joinpath(f).is_file() for f in ["data.csv", "features.json", "labels.json", "names.json"]):
                    with i.joinpath("labels.json").open() as f:
                        l.update(json.load(f))
                    # if so, move to the dataset's "files" folder
                    i = i.joinpath("files")
                for f in i.listdir(is_executable):
                    f = str(f)
                    if f not in e:
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
        r, kwargs['silent'] = [], False
        for exe in e:
            kwargs['label'] = l.get(Path(exe).stem, -1)
            r.append(method(self, exe, *args, **kwargs))
            kwargs['silent'] = True
        return r[0] if len(r) == 1 else r
    return _wrapper


def highlight_best(table_data, from_col=2):
    """ Highlight the highest values in the given table. """
    new_data = [table_data[0]]
    maxs = len(new_data[0][from_col:]) * [0]
    # search for best values
    for data in table_data[1:]:
        for i, value in enumerate(data[from_col:]):
            maxs[i] = max(maxs[i], float(value))
    # reformat the table, setting bold text for best values
    for data in table_data[1:]:
        new_row = data[:from_col]
        for i, value in enumerate(data[from_col:]):
            new_row.append(bold(value) if float(value) == maxs[i] else value)
        new_data.append(new_row)
    return new_data


# based on: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, ins, typ):
        return (super().__get__ if ins is None else self.__func__.__get__)(ins, typ)


class Config(ConfigParser):
    """ Simple Config class for handling some packing-box settings. """
    DEFAULTS = {
        'main': {
            'workspace': ("~/.packing-box", lambda v: Path(v, create=True, expand=True).absolute()),
        }
    }
    
    def __init__(self):
        super(Config, self).__init__()
        self.path = ConfigPath("packing-box", file=True)
        if not self.path.exists():
            self.path.touch()
        # get options from the target file
        self.read(str(self.path))
        # complete with default option-values
        try:
            sections = list(self.sections())
        except AttributeError:
            sections = []
        for section, options in self.DEFAULTS.items():
            if section not in sections:
                self.add_section(section)
            for opt, val in options.items():
                try:
                    val, func = val
                except ValueError:
                    pass
                s = super().__getitem__(section)
                if opt not in s:
                    s[opt] = val
    
    def __getitem__(self, option):
        for name in self.sections():
            sec = super().__getitem__(name)
            if option in sec:
                o = Config.DEFAULTS[name][option]
                return (o[1] if isinstance(o, tuple) and len(o) > 1 else str)(sec[option])
        raise KeyError(option)
    
    def __iter__(self):
        for section in self.sections():
            for option in super().__getitem__(section).keys():
                yield option
    
    def __setitem__(self, option, value):
        for section in self.sections():
            s = super().__getitem__(section)
            if option in s:
                s[option] = str(value)
                return
        raise KeyError(option)
    
    def items(self):
        for opt in sorted(x for x in self):
            yield opt, self[opt]
    
    def iteroptions(self):
        opts = []
        for name in self.sections():
            sec = super().__getitem__(name)
            for opt, val in sec.items():
                o = Config.DEFAULTS[name][opt]
                opts.append((opt, o[1] if isinstance(o, tuple) and len(o) > 1 else str, val))
        for o, v, f in sorted(opts, key=lambda x: x[0]):
            yield o, v, f
    
    def save(self):
        with self.path.open('w') as f:
            self.write(f)
config = Config()

