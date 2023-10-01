# -*- coding: UTF-8 -*-
from tinyscript import logging

from ...helpers import dict2, expand_formats, get_data, get_format_group, load_yaml_config, MetaBase, FORMATS


__all__ = ["Alterations"]


class Alteration(dict2):
    def __call__(self, executable, namespace, **kwargs):
        self._exe = executable
        # loop the specified number of times or simply once if not specified
        for _ in range(self.loop):
            parsed = executable.parse(self.parser)
            namespace.update({'binary': parsed, executable.group: parsed})
            self._logger.debug("applying alteration...")
            super().__call__(namespace)(parsed, self._logger)
            self._logger.debug("rebuilding binary...")
            self._logger.debug(" > build config: %s" % parsed._build_config)
            parsed.build()
    
    # default values on purpose not set via self.setdefault(...)
    @cached_property
    def apply(self):
        return self.get('apply', False)
    
    @cached_property
    def loop(self):
        return self.get('loop', 1)
    
    # 'parser' parameter in the YAML config has precedence on the overall configured parser
    @cached_property
    def parser(self):
        return self.get('parser', config['%s_parser' % self._exe.shortgroup])


class Alterations(list, metaclass=MetaBase):
    """ This class parses the YAML definitions of alterations to be applied to executables. It works as a list that
         contains the names of alterations applied to the executable given an input executable. """
    namespaces = None
    registry   = None
    
    @logging.bindLogger
    def __init__(self, exe=None, select=None, warn=False):
        a = Alterations
        # parse YAML alterations definition once
        if a.registry is None:
            src = a.source  # WARNING! this line must appear BEFORE a.registry={} because the first time that the
                            #           source attribute is called, it is initialized and the registry is reset to None
            a.namespaces, a.registry = {}, {}
            # collect properties that are applicable for all the alterations
            for name, params in load_yaml_config(src):
                r = params.pop('result', {})
                # consider most specific alterations first, then those for intermediate format classes and finally the
                #  collapsed class "All"
                for flist in [expand_formats("All"), [f for f in FORMATS.keys() if f != "All"], ["All"]]:
                    for fmt in flist:
                        expr = r.get(fmt) if isinstance(r, dict) else str(r)
                        if expr:
                            alt = Alteration(params, name=name, result=expr, logger=self.logger)
                            for subfmt in expand_formats(fmt):
                                a.registry.setdefault(subfmt, {})
                                a.registry[subfmt][alt.name] = alt
        # check the list of selected alterations if relevant, and filter out bad names (if warn=True)
        for name in (select or [])[:]:
            if name not in a.registry[exe.format]:
                msg = "Alteration '%s' does not exist" % name
                if warn:
                    self.logger.warning(msg)
                    select.remove(name)
                else:
                    raise ValueError(msg)
        if exe is not None:
            parser = None
            # prepare the namespace if not done yet for the target executable format
            if exe.format not in a.namespaces:
                from .modifiers import Modifiers
                # create the namespace for the given executable format (including the 'grid' helper defined hereafter)
                a.namespaces[exe.format] = {'grid': grid, 'logger': self.logger}
                # add constants specific to the target executable format
                a.namespaces[exe.format].update(get_data(exe.format))
                # add format-related data and modifiers
                a.namespaces[exe.format].update(Modifiers()[exe.group])
                # add format-specific alterations
                a.namespaces[exe.format].update(a.registry[exe.format])
            for name, alteration in a.registry[exe.format].items():
                if select is None and not alteration.apply or select is not None and name not in select:
                    continue
                # run the alteration given the format-specific namespace
                try:
                    alteration(exe, a.namespaces[exe.format])
                    self.logger.debug("applied alteration '%s'" % alteration.name)
                    self.append(name)
                except NotImplementedError as e:
                    self.logger.warning("'%s' is not supported yet for parser '%s'" % (e.args[0], exe.parsed.parser))
                except Exception as e:
                    self.logger.exception(e)
    
    @staticmethod
    def names(format="All"):
        Alterations(None)  # force registry initialization
        l = []
        for c in expand_formats(format):
            l.extend(list(Alterations.registry.get(c, {}).keys()))
        return sorted(list(set(l)))


def grid(modifier, params_grid, **eval_data):
    """ Run another modifier multiple times with different parameters.
    
    :param modifier:    modifier function
    :param params_grid: list of dictionnaries containing the parameter values
    """
    def _wrapper(parser=None, executable=None, **kw):
        d, e = {}, executable
        d.update(Alterations.namespaces[e.format])
        d.update(eval_data)
        for params in params_grid:
            parse_executable(parser, e, d)
            d.update(params)
            parser = modifier(d, parser=parser, executable=e, **kw)
            for p in params.keys():
                del d[p]
        return parser
    return _wrapper

