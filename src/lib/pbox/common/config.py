# -*- coding: UTF-8 -*-
import mdv
from tinyscript import configparser, re
from tinyscript.helpers import slugify, ConfigPath, Path
from tinyscript.report import *


__all__ = ["check_name", "config", "LABELS_BACK_CONV", "LOG_FORMATS", "NOT_LABELLED", "NOT_PACKED",
           "PACKING_BOX_SOURCES", "RENAME_FUNCTIONS"]

LOG_FORMATS = ["%(asctime)s [%(levelname)s] %(message)s", "%(asctime)s [%(levelname)-8s] %(name)-18s - %(message)s"]
PACKING_BOX_SOURCES = {
    'ELF': ["/usr/bin", "/usr/sbin"],
    'PE':  ["~/.wine32/drive_c/windows", "~/.wine64/drive_c/windows"],
}
PBOX_HOME = Path("~/.packing-box", expand=True)
RENAME_FUNCTIONS = {
    'as-is':   lambda p: p,
    'slugify': slugify,
}

# label markers and conversion for Scikit-Learn and Weka
NOT_LABELLED, NOT_PACKED = "?-"  # impose markers for distinguishing between unlabelled and not-packed data
LABELS_BACK_CONV = {NOT_LABELLED: -1, NOT_PACKED: None}  # values used with Scikit-learn for unlabelled and null class

_bl = lambda v: str(v).lower() in ["1", "true", "y", "yes"]
_np = lambda v: Path(str(v), create=True, expand=True).absolute()
_rp = lambda v: Path(str(v), expand=True).absolute()
_ws = lambda s, v: Path(s['workspace'].joinpath(v), create=True, expand=True).absolute()


def check_name(name, raise_error=True):
    """ Helper function for checking valid names according to the naming convention. """
    if name in [x for y in Config.DEFAULTS.values() for x in y.keys()] or name == "all" or \
       not re.match(r"(?i)^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$", name.basename if isinstance(name, Path) else str(name)):
        if raise_error:
            raise ValueError("Bad input name (%s)" % name)
        else:
            return False
    return name if raise_error else True


class Config(configparser.ConfigParser):
    """ Simple Config class for handling some packing-box settings. """
    DEFAULTS = {
        'main': {
            'workspace':    (PBOX_HOME, "PATH", "path to the workspace", _np, ["experiment"]),
            'experiments':  ("/mnt/share/experiments", "PATH", "path to the experiments folder", _np),
            'keep_backups': ("true", "BOOL", "keep backups of datasets ; for commands that trigger backups", _bl),
            'exec_timeout': ("10", "SECONDS", "execution timeout of items (detectors, packers, ...)", int),
        },
        'definitions': {
            'algorithms': ("~/.opt/algorithms.yml", "PATH", "path to the algorithms' YAML definition", _rp),
            'analyzers':  ("~/.opt/analyzers.yml",  "PATH", "path to the analyzers' YAML definition", _rp),
            'detectors':  ("~/.opt/detectors.yml",  "PATH", "path to the detectors' YAML definition", _rp),
            'features':   ("~/.opt/features.yml",   "PATH", "path to the features' YAML definition", _rp),
            'modifiers':  ("~/.opt/modifiers.yml",  "PATH", "path to the modifiers' YAML definition", _rp),
            'packers':    ("~/.opt/packers.yml",    "PATH", "path to the packers' YAML definition", _rp),
            'unpackers':  ("~/.opt/unpackers.yml",  "PATH", "path to the unpackers' YAML definition", _rp),
        },
        'logging': {
            'wine_errors': ("false", "BOOL", "display Wine errors", _bl),
        },
    }
    ENVVARS = ["experiment", "experiments"]
    HIDDEN = {
        'datasets': ("datasets", None, "", _ws),
        'models':   ("models",   None, "", _ws),
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
                func = None
                if len(val) == 4:
                    val, mvar, help, func = val
                elif len(val) == 5:
                    val, mvar, help, func, _ = val
                s = super().__getitem__(section)
                if opt not in s:
                    s[opt] = str(val if func is None else func(val))
    
    def __delitem__(self, option):
        if option in self.ENVVARS:
            PBOX_HOME.joinpath(option + ".env").remove(False)
        for section in self.sections():
            sec = super().__getitem__(section)
            if option in sec:
                del sec[option]
                return
        if option not in self.ENVVARS:
            raise KeyError(option)
    
    def __getitem__(self, option):
        for section in self.sections():
            sec = super().__getitem__(section)
            if option in sec:
                o = self.DEFAULTS[section][option]
                if isinstance(o, tuple) and len(o) > 4:
                    for override in o[4]:
                        v = config[override]
                        if v not in [None, ""]:
                            return o[3](v)
                if option in self.ENVVARS:
                    envf = PBOX_HOME.joinpath(option + ".env")
                    if envf.exists():
                        with envf.open() as f:
                            v = f.read().strip()
                        if v != "":
                            self[option] = v
                return (o[3] if isinstance(o, tuple) and len(o) > 3 else str)(sec[option])
        h = self.HIDDEN
        if option in h:
            v = h[option]
            return v[3](self, v[0]) if isinstance(v, tuple) else v
        if option in self.ENVVARS:
            envf = PBOX_HOME.joinpath(option + ".env")
            if envf.exists():
                with envf.open() as f:
                    return f.read().strip()
            return ""
        raise KeyError(option)
    
    def __iter__(self):
        for section in self.sections():
            for option in super().__getitem__(section).keys():
                yield option
    
    def __setitem__(self, option, value):
        if option in self.ENVVARS:
            with PBOX_HOME.joinpath(option + ".env").open("w") as f:
                f.write(str(value))
        for section in self.sections():
            s = super().__getitem__(section)
            if option in s:
                try:
                    _, func = self.DEFAULTS[section][option]
                except ValueError:
                    func = lambda x: x
                s[option] = str(func(value))
                return
        if option not in self.ENVVARS:
            raise KeyError(option)
    
    def items(self):
        for option in sorted(x for x in self):
            yield option, self[option]
    
    def iteroptions(self):
        options = []
        for section in self.sections():
            for option, value in super().__getitem__(section).items():
                o = self.DEFAULTS[section][option]
                options.append((option, o[3] if len(o) > 3 else str, value, o[1], o[2]))
        for o, f, v, m, h in sorted(options, key=lambda x: x[0]):
            yield o, f, v, m, h
    
    def overview(self):
        r = []
        for name in self.sections():
            sec = super().__getitem__(name)
            r.append(Section(name.capitalize()))
            mlen = 0
            for opt in sec.keys():
                mlen = max(mlen, len(opt))
            l = []
            for opt, val in sec.items():
                l.append("%s = %s" % (opt.ljust(mlen), str(val)))
            r.append(List(l))
        print(mdv.main(Report(*r).md()))
    
    def save(self):
        with self.path.open('w') as f:
            self.write(f)


config = Config()

