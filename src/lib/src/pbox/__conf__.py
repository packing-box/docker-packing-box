# -*- coding: UTF-8 -*-
import builtins as bi
from tinyscript import hashlib, logging
from tinyscript.helpers import slugify, Path
from warnings import filterwarnings

from .helpers.config import Config


filterwarnings("ignore", "Trying to unpickle estimator DecisionTreeClassifier")
filterwarnings("ignore", "Behavior when concatenating bool-dtype and numeric-dtype arrays is deprecated")


bi.null_logger = logging.nullLogger

bi.LOG_FORMATS = ["%(asctime)s [%(levelname)s] %(message)s", "%(asctime)s [%(levelname)-8s] %(name)-18s - %(message)s"]
bi.PACKING_BOX_SOURCES = {
    'ELF': ["/usr/bin", "/usr/sbin"],
    'PE':  ["~/.wine32/drive_c/windows", "~/.wine64/drive_c/windows"],
}
bi.PBOX_HOME = Path("~/.packing-box", create=True, expand=True)
for sf in ["conf", "data", "datasets", "models"]:
    PBOX_HOME.joinpath(sf).mkdir(0o764, True, True)
bi.RENAME_FUNCTIONS = {
    'as-is':   lambda p: p,
    'slugify': slugify,
}

# label markers and conversion for Scikit-Learn and Weka
bi.NOT_LABELLED, bi.NOT_PACKED = "?-"  # impose markers for distinguishing between unlabelled and not-packed data
bi.LABELS_BACK_CONV = {NOT_LABELLED: -1, NOT_PACKED: 0}  # values used with sklearn for unlabelled and null class


_bl = lambda s, v: str(v).lower() in ["1", "true", "y", "yes"]
_it = lambda s, v: int(v)
_np = lambda s, v: Path(str(v), create=True, expand=True).absolute()
_rp = lambda s, v: Path(str(v), expand=True).absolute()
_ws = lambda s, v: Path(s['workspace'].joinpath(v), create=True, expand=True).absolute()

opt_tuple = lambda k: ("conf/%s.yml" % k, "PATH", "path to %s YAML definition" % k, _rp, ["workspace", PBOX_HOME], True)


def _valid_hash(config, hash_algo):
    if hash_algo not in hashlib.algorithms_available:
        raise ValueError("Hash algorithm '%s' is not available in hashlib" % hash_algo)
    return hash_algo


bi.config = Config("packing-box",
    # defaults
    {
        'main': {
            'workspace':     (PBOX_HOME, "PATH", "path to the workspace", _np, ["experiment"]),
            'experiments':   ("/mnt/share/experiments", "PATH", "path to the experiments folder", _np),
            'keep_backups':  ("true", "BOOL", "keep backups of datasets ; for commands that trigger backups", _bl),
            'exec_timeout':  ("10", "SECONDS", "execution timeout of items (detectors, packers, ...)", _it),
            'cache_entries': ("1048576", "ENTRIES", "number of parsed samples in LRU cache", _it),
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
            'hash_algorithm': ("sha256", "ALGORITHM", "hashing algorithm for identifying samples", _valid_hash),
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

