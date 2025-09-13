# -*- coding: UTF-8 -*-
import builtins as bi
from functools import cached_property, lru_cache
from tinyscript import hashlib, logging
from tinyscript.helpers import classproperty, positive_float, positive_int, set_exception, slugify, Path
from warnings import filterwarnings, simplefilter

from .constants import *
from .helpers.config import *


filterwarnings("ignore", "Trying to unpickle estimator DecisionTreeClassifier")
filterwarnings("ignore", "Behavior when concatenating bool-dtype and numeric-dtype arrays is deprecated")
filterwarnings("ignore", "Cluster 1 is empty!")
filterwarnings("ignore", "Unable to import Axes3D.")
simplefilter("ignore", DeprecationWarning)
simplefilter("ignore", FutureWarning)
simplefilter("ignore", ResourceWarning)


bi.cached_property = cached_property
bi.cached_result = lru_cache
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
    if v not in ANGR_ENGINES:
        raise ValueError(f"bad Angr engine '{v}' ; shall be one of: {'|'.join(ANGR_ENGINES)}")
    return v
_ae.__name__ = "Angr engine"


def _bl(s, v):
    if (_v := str(v).lower()) in ["true", "yes", "1"]:
        return True
    elif _v in ["false", "no", "0"]:
        return False
    raise ValueError(v)
_bl.__name__ = "boolean"


def _cg(s, v):
    if v.lower() not in CFG_ALGORITHMS:
        raise ValueError(f"bad CFG algorithm '{v}' ; shall be one of: {'|'.join(CFG_ALGORITHMS)}")
    return v.capitalize()
_cg.__name__ = "CFG extraction algorithm"


def _cm(s, v):
    import matplotlib.pyplot as plt
    if v not in plt.colormaps():
        raise ValueError(f"invalid colormap '{v}'")
    return v

def _fmt(s, v):
    if v not in IMG_FORMATS:
        raise ValueError(f"invalid image format '{v}' ; shall be one of: {'|'.join(IMG_FORMATS)}")
    return v
_fmt.__name__ = "image format"


def _mg(s, v):
    if not .0 <= (v := positive_float(v)) <= .5:
        raise ValueError(f"invalid margin '{v}' ; shall belong to [.0,.5]")
    return v


def _nj(s, v):
    if (v := positive_int(v)) > CPU_COUNT:
        logging.getLogger().debug(f"This computer has {CPU_COUNT} CPUs ; reducing number of jobs to this value")
        v = CPU_COUNT
    return v


def _rp(s, v):
    if not (v := Path(str(v), expand=True).absolute()).exists():
        e = ValueError(v)
        e.setdefault = True
        raise e
    return v
_rp.__name__ = "path"


def _sty(s, v):
    import matplotlib.pyplot as plt
    if v not in (l := plt.style.available + ["default"]):
        raise ValueError(f"invalid pyplot style '{v}' ; shall be one of: {'|'.join(l)}")
    return v
_sty.__name__ = "pyplot style"


def _vh(s, v):
    if v not in hashlib.algorithms_available:
        raise ValueError(f"'{v}' is not a valid hash algorithm")
    return v
_vh.__name__ = "hash algorithm"


def _vs(s, v):
    if v not in ["mrsh_v2", "sdhash", "ssdeep", "tlsh"]:
        raise ValueError(f"'{v}' is not a valid similiraty algorithm")
    return v
_vs.__name__ = "similarity algorithm"


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
            'number_jobs':   ("6", "JOBS", "number of jobs to be run in parallel", _nj),
        },
        'cfg': {
            'angr_engine':            ("default", "ENGINE", "set the engine for CFG extraction by Angr", _ae),
            'depth_max_iterations':   ("1024", "ITERATIONS", "maximum iterations when determining graph depth", _it),
            'exclude_duplicate_sigs': ("true", "BOOL", "exclude node whose signature is a duplicate", _bl),
            'extract_algorithm':      ("emulated", "ALGO", "CFG extraction algorithm", _cg),
            'extract_timeout':        ("17", "SECONDS", "execution timeout for computing CFG of an executable", _it),
            'include_cut_edges':      ("true", "BOOL", "include the stored information about edges that got cut", _bl),
            'only_opcodes':           ("true", "BOOL", "store only the opcodes of each instruction in the node", _bl),
            'opcode_mnemonics':       ("false", "BOOL", "if only opcodes, store the opcode bytes or mnemonics", _bl),
            'store_loop_cut_info':    ("true", "BOOL", "keep loop cut information in node instances", _bl),
        },
        'definitions': {k: opt_tuple(k) for k in \
             ['algorithms', 'alterations', 'analyzers', 'detectors', 'features', 'packers', 'references', 'scenarios',
              'unpackers']},
        'logging': {
            'lief_logging': ("false", "BOOL", "display LIEF logging messages", _bl),
            'wine_errors':  ("false", "BOOL", "display Wine errors", _bl),
        },
        'others': {
            'autocommit':           ("false", "BOOL", "auto-commit in commands.rc (only works when experiment opened)",
                                     _bl),
            'data':                 ("data", "PATH", "path to executable formats' related data, relative to the "
                                     "workspace", _rp, ["workspace", PBOX_HOME], True),
            'file_balance_margin':  (".2", "MARGIN", "margin for the balance of file-related fields in a dataset "
                                     "(belongs to [.0,.5])", _mg),
            'label_balance_margin': (".1", "MARGIN", "margin for the balance of labels in a dataset "
                                     "(belongs to [.0,.5])", _mg),
            'hash_algorithm':       ("sha256", "ALGORITHM", "hashing algorithm for identifying samples", _vh),
            'fuzzy_hash_algorithm': ("ssdeep", "ALGORITHM", "algorithm for computing samples' fuzzy hash", _vs),
            'min_str_len':          ("4", "LENGTH", "minimal string length", _it),
            'vt_api_key':           ("", "API_KEY", "VirusTotal's RESTful API key"),
        },
        'parsers': {
            'default_parser': ("lief", "PARSER", "name of the module for parsing any format of executable"),
            'elf_parser':     psr_tuple("ELF"),
            'macho_parser':   psr_tuple("Mach-O"),
            'pe_parser':      psr_tuple("PE"),
        },
        'visualization': {
            'bbox_inches':     ("tight", "BBOX", "bbox in inches for saving the figure"),
            'colormap_main':   ("RdYlGn_r", "CMAP", "name of matplotlib.colors.Colormap to apply to plots", _cm),
            'colormap_other':  ("jet", "CMAP", "name of matplotlib.colors.Colormap to apply to plots", _cm),
            'dpi':             ("300", "DPI", "figures' dots per inch", _it),
            'font_family':     ("serif", "FAMILY", "font family for every text"),
            'font_size':       ("10", "SIZE", "base font size", _it),
            'img_format':      ("png", "FORMAT", "image format for saving figures", _fmt),
            'style':           ("default", "STYLE", "name of the PyPlot style to apply to plots", _sty),
            'transparent':     ("false", "BOOL", "save figure with transparent background", _bl),
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

