# -*- coding: UTF-8 -*-
import mdv
from tinyscript import configparser
from tinyscript.helpers import ConfigPath, Path
from tinyscript.report import *


__all__ = ["config", "LABELS_BACK_CONV", "LOG_FORMATS", "NAMING_CONVENTION", "NOT_LABELLED", "NOT_PACKED",
           "PACKING_BOX_SOURCES"]

LOG_FORMATS = ["%(asctime)s [%(levelname)s] %(message)s", "%(asctime)s [%(levelname)-8s] %(name)-18s - %(message)s"]
NAMING_CONVENTION = r"(?i)^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$"
PACKING_BOX_SOURCES = {
    'ELF': ["/sbin", "/usr/bin"],
    'PE':  ["~/.wine32/drive_c/windows", "~/.wine64/drive_c/windows"],
}

# label markers and conversion for Scikit-Learn and Weka
NOT_LABELLED, NOT_PACKED = "?-"  # impose markers for distinguishing between unlabelled and not-packed data
LABELS_BACK_CONV = {NOT_LABELLED: -1, NOT_PACKED: None}  # values used with Scikit-learn for unlabelled and null class

_rp = lambda v: Path(str(v), expand=True).absolute()
_ws = lambda s, v: Path(s['workspace'].joinpath(v), create=True, expand=True).absolute()


class Config(configparser.ConfigParser):
    """ Simple Config class for handling some packing-box settings. """
    DEFAULTS = {
        'main': {
            'workspace': ("~/.packing-box", lambda v: Path(str(v), create=True, expand=True).absolute()),
        },
        'definitions': {
            'algorithms': ("~/.opt/algorithms.yml", _rp),
            'analyzers':  ("~/.opt/analyzers.yml", _rp),
            'detectors':  ("~/.opt/detectors.yml", _rp),
            'features':   ("~/.opt/features.yml", _rp),
            'modifiers':  ("~/.opt/modifiers.yml", _rp),
            'packers':    ("~/.opt/packers.yml", _rp),
            'unpackers':  ("~/.opt/unpackers.yml", _rp),
        },
        'logging': {
            'wine_errors': ("false", lambda v: str(v).lower() in ["1", "true", "y", "yes"]),
        },
    }
    HIDDEN = {
        'datasets': ("datasets", _ws),
        'models':   ("models", _ws),
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
                    func = None
                s = super().__getitem__(section)
                if opt not in s:
                    s[opt] = val if func is None else str(func(val))
    
    def __getitem__(self, option):
        for section in self.sections():
            sec = super().__getitem__(section)
            if option in sec:
                o = Config.DEFAULTS[section][option]
                return (o[1] if isinstance(o, tuple) and len(o) > 1 else str)(sec[option])
        h = self.HIDDEN
        if option in h:
            v = h[option]
            return v[1](self, v[0]) if isinstance(v, tuple) else v
        raise KeyError(option)
    
    def __iter__(self):
        for section in self.sections():
            for option in super().__getitem__(section).keys():
                yield option
    
    def __setitem__(self, option, value):
        for section in self.sections():
            s = super().__getitem__(section)
            if option in s:
                try:
                    _, func = Config.DEFAULTS[section][option]
                except ValueError:
                    func = lambda x: x
                s[option] = str(func(value))
                return
        raise KeyError(option)
    
    def items(self):
        for option in sorted(x for x in self):
            yield option, self[option]
    
    def iteroptions(self):
        options = []
        for section in self.sections():
            for option, value in super().__getitem__(section).items():
                o = Config.DEFAULTS[section][option]
                options.append((option, o[1] if isinstance(o, tuple) and len(o) > 1 else str, value))
        for o, f, v in sorted(options, key=lambda x: x[0]):
            yield o, f, v
    
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

