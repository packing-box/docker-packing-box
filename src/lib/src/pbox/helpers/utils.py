# -*- coding: UTF-8 -*-
from tinyscript import functools, re
from tinyscript.helpers.common import lazy_load_module, lazy_object


def set_font(*a):
    plt.rcParams['font.family'] = "serif"

lazy_load_module("matplotlib", alias="mpl")
lazy_load_module("matplotlib.pyplot", alias="plt", postload=set_font)
lazy_load_module("numpy", alias="np")
lazy_load_module("yaml")


__all__ = ["at_interrupt", "benchmark", "bin_label", "bold", "class_or_instance_method", "execute_and_get_values_list",
           "get_counts", "lazy_object", "lazy_load_module", "mpl", "np", "plt", "shorten_str", "strip_version", "yaml",
           "COLORMAP"]

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


bin_label = lambda l: {NOT_LABELLED.lower(): -1, 'false': 0, NOT_PACKED.lower(): 0, 'true': 1, None: None} \
                      .get(l.lower(), 1)
bold = lambda text: "\033[1m{}\033[0m".format(text)
get_counts = lambda metadata, packed=True: {k: v for k, v in metadata['counts'].items() if k not in \
                                            ([NOT_LABELLED, NOT_PACKED] if packed else [NOT_LABELLED])}


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

