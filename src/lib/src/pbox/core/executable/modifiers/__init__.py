# -*- coding: UTF-8 -*-
"""
Conventions:
 - modifiers are decorators holding wrappers applied to a parsed executable given a namespace ; the parser is NOT
    selected here, it is 'lief' by default and can be tuned through ~/.packing-box.conf (global default parser) or
    alterations.yml using the 'parser' parameter (takes the precedence then)
   NB: wrappers can be themselves decorated with the supported_parsers(...) function to restrict their scope to only
        specific parsers they are compatible with
 - modifers' wrappers have keyword-arguments only and do not return anything ; if a new build is required, build
    instructions shall be set in the parsed._build_config dictionary
 - modifiers are named starting with a verb and describe the action performed
"""
from .elf import *
from .elf import __all__ as _elf
from .macho import *
from .macho import __all__ as _macho
from .pe import *
from .pe import __all__ as _pe
from ....helpers import format_shortname, FORMATS


__all__ = ["Modifiers"]


class Modifiers(dict):
    """ This class represents the dictionary of modifiers available per executable format. """
    def __init__(self):
        # load modifiers once
        if not getattr(self, "_init", False):
            for group in FORMATS.keys():
                if group == "All":
                    continue
                scope = "_" + format_shortname(group)  # i.e. _elf, _macho, _pe
                data_func = "get%s_data" % scope
                # set extra format-specific data first (derived from data files coming from ~/.packing-box/data/)
                self[group] = globals().get(data_func, lambda: {})()
                # then add format group modifiers
                self[group].update({k: globals()[k] for k in globals()[scope] if k != data_func})
            self._init = True

