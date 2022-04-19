# -*- coding: UTF-8 -*-
from tinyscript import configparser
from tinyscript.helpers import ConfigPath, Path


__all__ = ["config", "LOG_FORMATS"]

LOG_FORMATS = ["%(asctime)s [%(levelname)s] %(message)s", "%(asctime)s [%(levelname)-8s] %(name)-16s - %(message)s"]


_ws = lambda s, v: Path(s['workspace'].joinpath(v), create=True, expand=True).absolute()


class Config(configparser.ConfigParser):
    """ Simple Config class for handling some packing-box settings. """
    DEFAULTS = {
        'main': {
            'workspace': ("~/.packing-box", lambda v: Path(str(v), create=True, expand=True).absolute()),
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
                except TypeError:
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

