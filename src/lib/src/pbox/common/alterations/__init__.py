# -*- coding: UTF-8 -*-
from tinyscript import logging, re

from .elf import *
from .elf import __all__ as _elf
from .macho import *
from .macho import __all__ as _macho
from .pe import *
from .pe import __all__ as _pe

from .parsers import parse_executable
from ..data import get_data
from ...common.utils import *

lazy_load_module("yaml")


__all__ = ["Alterations"]


class Alteration(dict2):
    _fields = {'apply': True, 'loop': None, 'force_build': False, 'parameters': {}}
    
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
            # open the target YAML-formatted alterations set only once
            with open(a.source) as f:
                alterations = yaml.load(f, Loader=yaml.Loader) or {}
            a.namespaces, a.registry = {}, {}
            # collect properties that are applicable for all the alterations
            data_all = alterations.pop('defaults', {})
            for name, params in alterations.items():
                for i in data_all.items():
                    params.setdefault(*i)
                r = params.pop('result', {})
                # consider most specific alterations first, then those for intermediate format classes and finally the
                #  collapsed class "All"
                for flist in [expand_formats("All"), [f for f in FORMATS.keys() if f != "All"], ["All"]]:
                    for fmt in flist:
                        expr = r.get(fmt) if isinstance(r, dict) else str(r)
                        if expr:
                            a = Alteration(params, name=name, parent=self, result=expr)
                            for subfmt in expand_formats(fmt):
                                a.registry.setdefault(subfmt, {})
                                a.registry[subfmt][a.name] = a
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
                # add constants specific to the target executable format
                a.namespaces[exe.format] = get_data(exe.format)
                # add format-related helpers and other stuffs
                fmt_scope = "_" + format_shortname(get_format_group(exe.format))  # i.e. _elf, _macho, _pe
                a.namespaces[exe.format].update({k: globals()[k] for k in globals()[fmt_scope]})
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
                    self.logger.warning("%s: %s" % (name, str(e)))
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
            l.extend(list(Alterations.registry[c].keys()))
        return sorted(list(set(l)))

