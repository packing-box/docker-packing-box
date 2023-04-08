# -*- coding: UTF-8 -*-
import yaml
from tinyscript import logging, re

from .__common__ import *
from .__common__ import __all__ as __common__
from .elf import *
from .elf import __all__ as __elf__
from .macho import *
from .macho import __all__ as __macho__
from .pe import *
from .pe import __all__ as __pe__
from ...common.config import config
from ...common.utils import dict2, expand_formats, FORMATS


__all__ = ["Modifiers"]


class Modifier(dict2):
    _fields = {'apply': True}


class Modifiers(list):
    """ This class parses the YAML definitions of modifiers to be applied to executables for alterations.
         It works as a list that contains the names of alterations applied to the executable given in input. """
    registry = None
    source   = config['modifiers']
    
    @logging.bindLogger
    def __init__(self, exe):
        # parse YAML modifiers definition once
        if Modifiers.registry is None:
            # open the target YAML-formatted modifiers set only once
            with open(Modifiers.source) as f:
                modifiers = yaml.load(f, Loader=yaml.Loader) or {}
            Modifiers.registry = {}
            # collect properties that are applicable for all the modifiers
            data_all = modifiers.pop('defaults', {})
            for name, params in modifiers.items():
                for i in data_all.items():
                    params.setdefault(*i)
                r = params.pop('result', {})
                # consider most specific modifiers first, then those for intermediate format classes and finally the
                #  collapsed class "All"
                for clist in [expand_formats("All"), list(FORMATS.keys())[1:], ["All"]]:
                    for c in clist:
                        expr = r.get(c) if isinstance(r, dict) else str(r)
                        if expr:
                            m = Modifier(params, name=name, parent=self, result=expr)
                            for c2 in expand_formats(c):
                                Modifiers.registry.setdefault(c2, {})
                                Modifiers.registry[c2][m.name] = m
        if exe is not None:
            parsed, parser = None, None
            for name, modifier in Modifiers.registry[exe.format].items():
                if not modifier.apply:
                    continue
                if parser is None or modifier.parser != parser:
                    parser = modifier.parser
                    parsed = parser(exe.realpath)
                d = {}
                d.update(__common__)
                md = __elf__ if exe.format in expand_formats("ELF") else \
                     __macho__ if exe.format in expand_formats("Mach-O") else\
                     __pe__ if exe.format in expand_formats("PE") else []
                d.update({k: globals()[k] for k in md})
                kw = {'executable': exe, 'parsed': parsed}
                try:
                    kw['sections'] = parsed.sections
                except:
                    pass
                try:
                    modifier(d, **kw)
                    self.append(name)
                except Exception as e:
                    self.logger.warning("%s: %s" % (name, str(e)))

