# -*- coding: UTF-8 -*-
from tinyscript import functools, re

lazy_load_module("json")
lazy_load_module("numpy", alias="np")
lazy_load_module("yaml")


__all__ = ["at_interrupt", "benchmark", "bin_label", "bold", "class_or_instance_method", "entropy",
           "execute_and_get_values_list", "get_counts", "json", "json_cache", "np", "pd", "shorten_str",
           "strip_version", "warn_once", "yaml"]

_WARN_CACHE = {}


bin_label = lambda l: {NOT_LABELLED.lower(): -1, 'false': 0, NOT_PACKED.lower(): 0, 'true': 1, None: None} \
                      .get(l.lower(), 1)
bold = lambda text: "\033[1m{}\033[0m".format(text)
get_counts = lambda metadata, packed=True: {k: v for k, v in metadata['counts'].items() if k not in \
                                            ([NOT_LABELLED, NOT_PACKED] if packed else [NOT_LABELLED])}


def __init_pd(module):
    def to_yml(self, path_or_buf=None, **kwargs):
        from _io import BufferedWriter
        d = json.loads(self.to_json(**kwargs))
        if path_or_buf is None:
            return "" if (y := yaml.dump(d)) == "{}\n" else y
        with (path_or_buf if isinstance(path_or_buf, BufferedWriter) else open(path_or_buf, 'wt')) as f:
            yaml.dump(d, f)
    module.DataFrame.to_yml = to_yml
lazy_load_module("pandas", alias="pd", postload=__init_pd)


def at_interrupt():
    """ Interrupt handler """
    logger.warn("Interrupted by the user.")


def benchmark(f):
    """ Decorator for benchmarking function executions. """
    from time import perf_counter, time
    def _wrapper(*args, **kwargs):
        t = perf_counter if kwargs.pop("perf", True) else time
        start = t()
        r = f(*args, **kwargs)
        dt = t() - start
        return r, dt
    return _wrapper


def entropy(data, k=256, max_entropy=False):
    if not 0 < k <= 256:
        raise ValueError("k shall belong to [1,256]")
    from collections import Counter
    from math import log2
    e, t, c = .0, len(data), Counter(data)
    for p in [n / t for _, n in c.most_common(k)]:
        e -= p * log2(p)
    return entropy(range(k)) if e == 0. and max_entropy else e


def execute_and_get_values_list(command, offset=1):
    """ Execute an OS command and parse its output considering lines of comma-separated values (ignoring values before
         the given offset).
        
        NB: This is especially used for running features extraction tools.
    """
    from ast import literal_eval
    from tinyscript.helpers import execute_and_log as run
    out, err, retc = run(command)
    if retc == 0:
        values = []
        for x in out.decode().strip().split(",")[offset:]:
            try:
                values.append(literal_eval(x))
            except ValueError:
                values.append(x)
        return values


def json_cache(name, key, force=False):
    cache = PBOX_HOME.joinpath("cache", name)
    cache.mkdir(parents=True, exist_ok=True)
    def _wrapper(f):
        @functools.wraps(f)
        def _subwrapper(*a, **kw):
            cache_file = cache.joinpath(f"{key}.json")
            if cache_file.exists():
                if force:
                    cache_file.remove()
                else:
                    with cache_file.open('rb') as fin:
                        return json.load(fin)
            r = f(*a, **kw)
            if r is not None and len(r) > 0:
                with cache_file.open('w') as fout:
                    try:
                        json.dump(r, fout, indent=2)
                    except TypeError:
                        from vt.object import UserDictJsonEncoder
                        fout.truncate()
                        fout.write(json.dumps(r, cls=UserDictJsonEncoder, indent=2))
            return r
        return _subwrapper
    return _wrapper


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


def warn_once(logger, message):
    """ Simple helper to ensure that the input logger warns the input message only once. """
    if logger is None:
        return
    if (k := id(logger)) not in _WARN_CACHE.keys():
        _WARN_CACHE.setdefault(k, set())
    if message not in _WARN_CACHE[k]:
        logger.warning(message)
        _WARN_CACHE[k].add(message)


# based on: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, ins, typ):
        return (super().__get__ if ins is None else self.__func__.__get__)(ins, typ)

