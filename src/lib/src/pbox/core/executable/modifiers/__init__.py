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
from ....helpers import format_shortname


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
                self[group].update({k.lstrip("_"): globals()[k] for k in _common + globals()[scope] if k != data_func})
            self._init = True


# --------------------------------------------- Common Modifiers -------------------------------------------------------
# Important note: given line 38, specific modifiers have precedence on common ones
_common = ["_add_section", "_rename_all_sections", "_rename_section"]


def _add_section(name, data=b""):
    """ Add a section. """
    def _add_section(parsed, logger):
        s = type(parsed)._section_class(name=name)
        s.content = list(data)
        parsed.add_section(s)
    return _add_section


def _rename_all_sections(old_sections, new_sections):
    """ Rename a given list of sections. """
    modifiers = [rename_section(old, new) for old, new in zip(old_sections, new_sections)]
    def _rename_all_sections(parsed, logger):
        for m in modifiers:
            try:
                parser = m(parsed, logger)
            except LookupError:
                parser = None
        return parser
    return _rename_all_sections


def _rename_section(old_section, new_name, error=True):
    """ Rename a given section. """
    if error and old_section is None:
        raise ValueError("Old section shall not be None")
    if new_name is None:
        raise ValueError("New section name shall not be None")
    def _rename_section(parsed, logger):
        try:
            sec = parsed.section(old_section, original=True) if isinstance(old_section, str) else old_section
        except ValueError:
            if error:
                raise
            return
        logger.debug(">> rename: %s -> %s" % (sec.name or "<empty>", new_name))
        sec.name = new_name
    return _rename_section

