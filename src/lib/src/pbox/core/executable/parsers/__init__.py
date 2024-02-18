# -*- coding: UTF-8 -*-
from importlib import import_module
from tinyscript.helpers import set_exception

from .__common__ import *
from .__common__ import __all__ as _common
from ....helpers.formats import get_format_group


__all__ = ["get_parser"] + _common

set_exception("KeyNotAllowedError", "KeyError")
set_exception("ParserError")
set_exception("SectionError")


def get_parser(parser, exe_format):
    """ Get a parser by name or by class. """
    try:
        m = import_module("." + parser, "pbox.core.executable.parsers")
    except ImportError:
        raise ImportError(f"no parsing module named '{parser}'")
    try:
        return getattr(m, exe_format, getattr(m, get_format_group(exe_format)))
    except AttributeError:
        raise ValueError(f"no parser available for format '{exe_format}'")

