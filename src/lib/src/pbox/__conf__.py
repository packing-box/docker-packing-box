# -*- coding: UTF-8 -*-
import builtins as bi
from functools import cached_property
from tinyscript import hashlib, logging
from tinyscript.helpers import classproperty, positive_int, slugify, Path
from warnings import filterwarnings, simplefilter

from .constants import *
from .helpers.config import *


filterwarnings("ignore", "Trying to unpickle estimator DecisionTreeClassifier")
filterwarnings("ignore", "Behavior when concatenating bool-dtype and numeric-dtype arrays is deprecated")
simplefilter("ignore", ResourceWarning)


bi.cached_property = cached_property
bi.classproperty = classproperty
bi.configure_logging = configure_logging
bi.null_logger = logging.nullLogger


_it = lambda s, v: positive_int(v)
_it.__name__ = "positive integer"
_np = lambda s, v: Path(str(v), create=True, expand=True).absolute()
_st = lambda s, v: str(v)
_ws = lambda s, v: Path(s['workspace'].joinpath(v), create=True, expand=True).absolute()


def _bl(s, v):
    _v = str(v).lower()
    if _v in ["true", "yes"]:
        return True
    elif _v in ["false", "no"]:
        return False
    raise ValueError(v)
_bl.__name__ = "boolean"


def _rp(s, v):
    v = Path(str(v), expand=True).absolute()
    if not v.exists():
        raise ValueError(v)
    return v
_rp.__name__ = "path"


def _vh(s, v):
    if v not in hashlib.algorithms_available:
        raise ValueError(v)
    return v
_vh.__name__ = "hash algorithm"


opt_tuple = lambda k: ("conf/%s.yml" % k, "PATH", "path to %s YAML definition" % k, _rp, ["workspace", PBOX_HOME], True)
psr_tuple = lambda f: (None, "PARSER", "name of the module for parsing %s executables" % f, _st, "default_parser")


for sf in ["conf", "data", "datasets", "models"]:
    PBOX_HOME.joinpath(sf).mkdir(0o764, True, True)

bi.config = Config("packing-box",
    # defaults
    {
        'main': {
            'workspace':     (PBOX_HOME, "PATH", "path to the workspace", _np, ["experiment"]),
            'experiments':   ("/mnt/share/experiments", "PATH", "path to the experiments folder", _np),
            'backup_copies': ("3", "COPIES", "keep N backups of datasets ; for commands that trigger backups", _it),
            'exec_timeout':  ("10", "SECONDS", "execution timeout of items (detectors, packers, ...)", _it),
        },
        'definitions': {k: opt_tuple(k) for k in \
                        ['algorithms', 'alterations', 'analyzers', 'detectors', 'features', 'packers', 'unpackers']},
        'logging': {
            'lief_errors': ("false", "BOOL", "display LIEF parsing errors", _bl),
            'wine_errors': ("false", "BOOL", "display Wine errors", _bl),
        },
        'others': {
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
    },
    # envvars
    {
        'experiment':  ("PATH", "path to the current experiment folder", _np),
        'experiments': ("PATH", "path to the experiments folder", _np),
    },
    # hidden
    {
        'datasets': ("datasets", None, "", _ws),
        'models':   ("models",   None, "", _ws),
    })

