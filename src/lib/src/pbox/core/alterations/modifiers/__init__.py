# -*- coding: UTF-8 -*-
from .elf import *
from .elf import __all__ as _elf
from .macho import *
from .macho import __all__ as _macho
from .pe import *
from .pe import __all__ as _pe
from ....helpers import format_shortname, FORMATS


__all__ = ["Modifiers"]


class Modifiers(dict):
    """ This class represents the dictionary of modifiers to be extracted for a given list of executable formats. """
    def __init__(self):
        # load modifiers once
        if not getattr(self, "_init", False):
            for group in FORMATS.keys():
                if group == "All":
                    continue
                scope = "_" + format_shortname(group)  # i.e. _elf, _macho, _pe
                data_func = "get%s_data" % scope
                # set extra format-specific data first (derived from format-specific data coming from ~/.opt/data/)
                self[group] = globals().get(data_func, lambda: {})()
                # then add format group modifiers
                self[group].update({k: globals()[k] for k in globals()[scope] if k != data_func})
            self._init = True

