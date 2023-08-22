# -*- coding: UTF-8 -*-
from functools import cached_property
from tinyscript import logging

from .parsers import parse_executable
from ...helpers import *


__all__ = ["Alterations"]


class Alteration(dict2):
    def __call__(self, namespace, parser=None, executable=None, **kwargs):
        # include alteration's parameters in the namespace for the computation
        namespace.update(self.parameters)
        # loop the specified number of times or simply once if self.loop not defined
        for _ in range(self.loop or 1):
            parse_executable(parser, executable, namespace)
            parser = super().__call__(namespace, parser=parser, executable=executable, **kwargs)
        # rebuild the target binary if needed
        if parser is not None:
            if self.force_build:
                parser.build()
                return
        # remove alteration's parameters from the namespace before returning
        for k in self.parameters.keys():
            del namespace[k]
        return parser
    
    @cached_property
    def apply(self):
        return self.get('apply', True)
    
    @cached_property
    def force_build(self):
        return self.get('force_build', False)
    
    @cached_property
    def loop(self):
        return self.get('loop')


class Alterations(list, metaclass=MetaBase):
    """ This class parses the YAML definitions of alterations to be applied to executables. It works as a list that
         contains the names of alterations applied to the executable given an input. """
    namespaces = None
    registry   = None
    
    @logging.bindLogger
    def __init__(self, exe, select=None, warn=False):
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
                # create the namespace for the given executable format
                a.namespaces[exe.format] = {'grid': grid}
                # add constants specific to the target executable format
                a.namespaces[exe.format].update(get_data(exe.format))
                # add format-related data and modifiers
                a.namespaces[exe.format].update(Modifiers()[get_format_group(exe.format)])
                # add format-specific alterations
                a.namespaces[exe.format].update(a.registry[exe.format])
            for name, alteration in a.registry[exe.format].items():
                if select is None and not alteration.apply or select is not None and name not in select:
                    continue
                # run the alteration given the format-specific namespace
                try:
                    parser = alteration(a.namespaces[exe.format], parser, exe)
                    self.append(name)
                except Exception as e:
                    self.logger.exception(e)
            # ensure the target executable is rebuilt
            if parser is not None:
                parser.build()
            # ensure the namespace is cleaned up from specific names
            for k in ["compute_checksum", "sections"]:
                del a.namespaces[exe.format][k]
    
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

