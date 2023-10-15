# -*- coding: UTF-8 -*-
from tinyscript import configparser, logging, re
from tinyscript.helpers import ConfigPath, Path


__all__ = ["configure_logging", "Config"]


_BASE_LOGGERS = [l for l in logging.root.manager.loggerDict] + ["rich"]
_NAMING_CONVENTION = r"(?i)^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$"


def configure_logging(verbose=False, levels=("INFO", "DEBUG")):
    logging.configLogger(logging.getLogger("main"), levels[verbose], fmt=LOG_FORMATS[min(verbose, 1)], relative=True)
    logging.setLoggers(*[l for l in logging.root.manager.loggerDict if l not in _BASE_LOGGERS])


class Config:
    """ Config class for getting settings from a base configuration file and handling them with special attributes.
         It is particular in that it enforces sections and options and does not allow any other options. It is aimed to
         edit options and save only those deviating from default values.
    
    :param name:     name of the application (i.e. on Linux, will be used to make ~/.[name]/ and ~/.[name].conf)
    :param defaults: dictionary (keys are config section names) of dictionaries (option name as the key and option
                      description as value ; see hereafter)
    :param envars:   list of environment variables to be retrieved from ~/.[name]/[envvar].env
    :param hidden:   dictionary of hidden (not user-configurable) options (same option format as for 'defaults')
    :param naming:   convention for option names (by default, hypen- or underscore-separated alphanumeric)
    
    Option description formats (apply to 'defaults' and 'hidden'):
      - 4-tuple: (default, metavar, description, transform_function)
      - 5-tuple: (...                                              , overrides)
      - 6-tuple: (...                                                         , join_default_to_override)
    """
    def __init__(self, name, defaults=None, envars=None, hidden=None, naming=_NAMING_CONVENTION):
        # keep a config object for holding changed options, to be saved to the destination path ;
        #  all other options are taken from defaults, environments variables or hidden options
        self.__config = configparser.ConfigParser()
        self._defaults, self._envars, self._hidden, self._naming = defaults or {}, envars or [], hidden or {}, naming
        self._path = ConfigPath(name, file=True)
        if not self._path.exists():
            self._path.touch()
        # get options from the target file
        self.__config.read(str(self._path))
    
    def __delitem__(self, option):
        """ Delete an item taking enviornment variables registered in .env files too. """
        if option in self.sections():
            raise ValueError("input name '%s' matches a section" % option)
        if option in self._envars:
            PBOX_HOME.joinpath(option + ".env").remove(False)
        for s in map(self.__config.__getitem__, self.__config.sections()):
            if option in s:
                del s[option]
                if len(s) == 0:
                    del s
        if option not in self._envars:
            raise KeyError(option)
    
    def __getitem__(self, option):
        """ Get an option's value, throwing an error if this option is not defined. """
        return self.get(option, error=True)
    
    def __iter__(self):
        """ Iterate over option names among the configuration sections. """
        for section, options in self._defaults.items():
            for option in options.keys():
                yield option
    
    def __setitem__(self, option, value):
        """ Set item writing an environment variable registered in a .env file if relevant. """
        if option in self._envars:
            with PBOX_HOME.joinpath(option + ".env").open("w") as f:
                f.write(str(value))
            return
        for s in self.sections():
            if option in self._defaults[s]:
                o = self._defaults[s][option]
                func = o[3] if len(o) >= 4 else lambda s, v, *a: str(v)
                value = func(self, value)
                # set option in the attached ConfigParser
                if self.__config.has_section(s):
                    self.__config[s][option] = str(value)
                    # if the new value equals the default, then it is useless to keep it defined
                    if self.get(option) == self.default(option):
                        del self.__config[s][option]
                        if len(self.__config[s]) == 0:
                            self.__config.remove_section(s)
                else:
                    self.__config.add_section(s)
                    self.__config[s][option] = str(value)
                return
        raise KeyError(option)
    
    def check(self, name, raise_error=True):
        """ Check option name against the naming convention. """
        if name in [x for y in self._defaults.values() for x in y.keys()] or name == "all" or \
           not re.match(self._naming, name.basename if isinstance(name, Path) else str(name)):
            if raise_error:
                raise ValueError("Bad input name (%s)" % name)
            else:
                return False
        return name if raise_error else True
    
    def default(self, option):
        """ Get the default value for an option. """
        for s in self.sections():
            if option in self._defaults[s].keys():
                o = self._defaults[s][option]
                # dummy option value, return as is
                if not isinstance(o, (list, tuple)):
                    return o
                # 3-tuple format: (default, metavar, description)
                default_value, metavar, description = o[:3]
                # 4-tuple format: (..., transform_function)
                func = o[3] if len(o) >= 4 else lambda s, v, *a: str(v)
                return func(self, default_value)
        raise KeyError(option)
    
    def func(self, option):
        """ Get the type function for an option. """
        for s in self.sections():
            if option in self._defaults[s].keys():
                o = self._defaults[s][option]
                return o[3] if len(o) >= 4 else lambda s, v, *a: str(v)
        raise KeyError(option)
    
    def get(self, option, default=None, sections=None, error=False):
        """ Get the value of an option, only considering the given list of sections (if None, consider all). """
        sections = list(self.sections()) if sections is None else \
                   [sections] if not isinstance(sections, list) else sections
        for s in sections:
            if s not in self.sections():
                raise ValueError("Bad section name '%s'" % s)
        # first, check for a hidden option (that cannot be set by the user)
        h = self._hidden
        if option in h:
            o = h[option]
            return o[3](self, o[0]) if isinstance(o, tuple) else o
        # second, check for an existing environment variable
        ev = self._envars
        if option in ev:
            envf = PBOX_HOME.joinpath(option + ".env")
            if envf.exists():
                with envf.open() as f:
                    v = f.read().strip()
                if v != "":
                    o = ev[option]
                    return o[2](self, v) if isinstance(o, tuple) else o
        # now, look at options from the bound ConfigParser or the registered defaults, given the input sections list to
        #  be considered
        for s in sections:
            if option not in self._defaults[s]:
                continue
            o = self._defaults[s][option]
            func = o[3] if len(o) >= 4 else lambda s, v, *a: str(v)
            # then, check for modified values (loaded from ~/.packing-box.conf) saved into self.__config
            if self.__config.has_section(s) and option in self.__config[s]:
                return func(self, self.__config[s][option])
            if isinstance(o, tuple) and len(o) > 4:
                # special 5-tuple format: (..., overrides)
                #  list of overrides gives the precedence to values from particular config keys if they are set ;
                #   e.g. ["experiment"] will cause checking if the 'experiment' config key is set first, then
                #        considering the default value
                if o[0] is not None:
                    chk = False
                    for override in o[4]:
                        try:
                            value = self[override] if isinstance(override, str) else override
                        except KeyError:
                            continue
                        # special 6-tuple format: (..., join_default_to_override)
                        #  if True, the default value will be joined to the override value
                        if len(o) > 5 and o[5]:
                            value, chk = value.joinpath(o[0]), o[5]
                        if value not in [None, ""] and (not chk or Path(value).exists()):
                            return func(self, value)
                # special 5-tuple format: (..., pointer_to_default)
                #  if default value is None, look at another option designated by pointer_to_default ;
                #   allows to define the option while setting no default so that it can still be edited by the user in
                #   the configuration file (~/.packing-box.conf)
                else:
                    return self.get(o[4])
            # finally, check for a default value
            try:
                return self.default(option)
            except KeyError:
                pass
        if error:
            raise KeyError(option)
        return default
    
    def items(self):
        """ Iterate over options, including the name and value. """
        for option in sorted(x for x in self):
            yield option, self.get(option)
    
    def iteroptions(self, sections=None):
        """ Iterate over options according to their definition attributes.
        
        Output format: (option_name, transform_function, value, metavar, description)
        """
        options = []
        for s in self.itersections(sections):
            for option in s.keys():
                o = self._defaults[s.name][option]
                yield option, o[3] if len(o) > 3 else lambda s, v: str(v), self.get(option), o[1], o[2]
    
    def itersections(self, sections=None):
        """ Iterate over section objects. """
        sections = self.sections() if sections is None else [sections] if not isinstance(sections, list) else sections
        for section in sections:
            yield self.section(section)
    
    def overview(self):
        """ Get a renderable overview of the configuration. """
        from tinyscript.report import List, Section
        from .rendering import render
        r = []
        for s in self.itersections():
            r.append(Section(s.name.capitalize()))
            mlen = max(map(len, s.keys()))
            r.append(List(list(map(lambda opt: "%s = %s" % (opt.ljust(mlen), self[opt]), s.keys()))))
        render(*r)
    
    def save(self):
        """ Write the current configuration to its path. """
        with self._path.open('w') as f:
            self.__config.write(f)
    
    def section(self, section):
        """ Create an orphan SectionProxy object with final values (taking environment variables, modified values and
             defaults into account). """
        s = Section(section)
        base = self._hidden if section == "hidden" else \
               self._envars if section == "envars" else \
               self._defaults[section] if section in self._defaults else \
               None
        if base is None:
            raise configparser.NoSectionError(section)
        for option in base.keys():
            s[option] = self.get(option)
        return s
    
    def sections(self):
        """ Iterate over section names (taken from the defaults). """
        for section in self._defaults.keys():
            yield section


class Section(dict):
    def __init__(self, name):
        self.name = name

