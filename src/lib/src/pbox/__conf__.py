# -*- coding: UTF-8 -*-
import builtins as bi
from functools import cached_property
from tinyscript import hashlib, logging
from tinyscript.helpers import classproperty, positive_int, set_exception, slugify, Path
from warnings import filterwarnings, simplefilter

from .constants import *
from .helpers.config import *


filterwarnings("ignore", "Trying to unpickle estimator DecisionTreeClassifier")
filterwarnings("ignore", "Behavior when concatenating bool-dtype and numeric-dtype arrays is deprecated")
simplefilter("ignore", DeprecationWarning)
simplefilter("ignore", FutureWarning)
simplefilter("ignore", ResourceWarning)


bi.cached_property = cached_property
bi.classproperty = classproperty
bi.configure_logging = configure_logging
bi.null_logger = logging.nullLogger
bi.set_exception = set_exception


_it = lambda s, v: positive_int(v)
_it.__name__ = "positive integer"
_np = lambda s, v: Path(str(v), create=True, expand=True).absolute()
_st = lambda s, v: str(v)
_ws = lambda s, v: Path(s['workspace'].joinpath(v), create=True, expand=True).absolute()


def _ae(s, v):
    if v not in ["default", "pcode", "vex"]:
        raise ValueError(f"bad Angr engine '{v}' ; shall be one of: default|pcode|vex")
    return v
_ae.__name__ = "Angr engine"


def _bl(s, v):
    _v = str(v).lower()
    if _v in ["true", "yes", "1"]:
        return True
    elif _v in ["false", "no", "0"]:
        return False
    raise ValueError(v)
_bl.__name__ = "boolean"


def _cg(s, v):
    if v.lower() not in ["emulated", "fast"]:
        raise ValueError(f"bad CFG algorithm '{v}' ; shall be one of: emulated|fast")
    return v.capitalize()
_cg.__name__ = "CFG extraction algorithm"


def _fmt(s, v):
    if v not in IMG_FORMATS:
        raise ValueError(f"invalid image format '{v}' ; shall be one of: {'|'.join(IMG_FORMATS)}")
    return v
_fmt.__name__ = "image format"


def _rp(s, v):
    v = Path(str(v), expand=True).absolute()
    if not v.exists():
        raise ValueError(v)
    return v
_rp.__name__ = "path"


def _sty(s, v):
    import matplotlib.pyplot as plt
    l = plt.style.available + ["default"]
    if v not in l:
        raise ValueError(f"invalid pyplot style '{v}' ; shall be one of: {'|'.join(l)}")
    return v
_sty.__name__ = "pyplot style"


def _vh(s, v):
    if v not in hashlib.algorithms_available:
        raise ValueError(f"'{v}' is not a valid hash algorithm")
    return v
_vh.__name__ = "hash algorithm"


opt_tuple = lambda k: (f"conf/{k}.yml", "PATH", f"path to {k} YAML definition", _rp, ["workspace", PBOX_HOME], True)
psr_tuple = lambda f: (None, "PARSER", f"name of the module for parsing {f} executables", _st, "default_parser")


for sf in ["conf", "data", "datasets", "models"]:
    PBOX_HOME.joinpath(sf).mkdir(0o764, True, True)

# Option description formats (apply to 'defaults' and 'hidden'):
#  - 4-tuple: (default, metavar, description, transform_function)
#  - 5-tuple: (...                                              , overrides)
#  - 6-tuple: (...                                                         , join_default_to_override)
bi.config = Config("packing-box",
    # defaults
    {
        'main': {
            'workspace':     (PBOX_HOME, "PATH", "path to the workspace", _np, ["experiment"]),
            'experiments':   ("/mnt/share/experiments", "PATH", "path to the experiments folder", _np),
            'backup_copies': ("3", "COPIES", "keep N backups of datasets ; for commands that trigger backups", _it),
            'exec_timeout':  ("20", "SECONDS", "execution timeout of items (detectors, packers, ...)", _it),
        },
        'cfg': {
            'angr_engine':       ("pcode", "ENGINE", "set the engine for CFG extraction by Angr", _ae),
            'extract_algorithm': ("fast", "ALGO", "CFG extraction algorithm", _cg),
            'extract_timeout':   ("20", "SECONDS", "execution timeout for computing CFG of an executable", _it),
        },
        'definitions': {k: opt_tuple(k) for k in \
             ['algorithms', 'alterations', 'analyzers', 'detectors', 'features', 'packers', 'scenarios', 'unpackers']},
        'logging': {
            'lief_logging': ("false", "BOOL", "display LIEF logging messages", _bl),
            'wine_errors': ("false", "BOOL", "display Wine errors", _bl),
        },
        'others': {
            'autocommit':     ("false", "BOOL", "auto-commit in commands.rc (only works when experiment opened)", _bl),
            'data':           ("data", "PATH", "path to executable formats' related data, relative to the workspace",
                               _rp, ["workspace", PBOX_HOME], True),
            'hash_algorithm': ("sha256", "ALGORITHM", "hashing algorithm for identifying samples", _vh),
        },
        'parsers': {
            'default_parser': ("lief", "PARSER", "name of the module for parsing any format of executable"),
            'elf_parser':     psr_tuple("ELF"),
            'macho_parser':   psr_tuple("Mach-O"),
            'pe_parser':      psr_tuple("PE"),
        },
        'visualization': {
            'bbox_inches':     ("tight", "BBOX", "bbox in inches for saving the figure"),
            #FIXME: enforce list of valid colormaps
            'colormap':        ("jet", "CMAP", "name of matplotlib.colors.Colormap to apply to plots"),
            'dpi':             ("200", "DPI", "figures' dots per inch", _it),
            'font_family':     ("serif", "FAMILY", "font family for every text"),
            'font_size':       ("10", "SIZE", "base font size", _it),
            'format':          ("png", "FORMAT", "image format for saving figures", _fmt),
            'style':           ("default", "STYLE", "name of the PyPlot style to apply to plots", _sty),
        },
    },
    # envvars
    {
        'autocommit':  ("BOOL", "auto-commit commands in commands.rc (only works when experiment opened)", _bl),
        'experiment':  ("PATH", "path to the current experiment folder", _np),
        'experiments': ("PATH", "path to the experiments folder", _np),
    },
    # hidden
    {
        'datasets': ("datasets", None, "", _ws),
        'models':   ("models",   None, "", _ws),
    })

