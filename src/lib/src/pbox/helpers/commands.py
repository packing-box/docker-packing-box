# -*- coding: UTF-8 -*-
from tinyscript import re, sys
from tinyscript.argreparse import ArgumentParser
from tinyscript.helpers import get_parsers, PathBasedDict as pbdict, Path, PythonPath
from tinyscript.parser import ProxyArgumentParser

from pprint import pprint

__all__ = ["get_commands"]


_COMMANDS = {}


def get_commands(tool, cond="", category=None, logger=None):
    if tool in _COMMANDS and cond in _COMMANDS[tool]:
        return _COMMANDS[tool][cond]
    if isinstance(tool, str):
        tool = Path(f"~/.opt/tools/{tool}", expand=True)
    parent, child, path, cmds, rm = None, None, (), pbdict(), []
    for parser in get_parsers(tool, cond=cond, logger=logger).values():
        if isinstance(parser, ArgumentParser):
            if parent == "main" and category is not None and getattr(parser, "category", None) != category:
                rm.append(parser.name)
            try:
                nparent, nchild = parser._parent._parent.name, parser.name
            except AttributeError:
                nparent, nchild = None, parser.name
            if nparent is None:
                continue
            # depth increases
            if nparent == child:
                path += (nchild, )
            # depth does not change
            elif nparent == parent:
                path = path[:-1] + (nchild, )
            # depth decreases
            elif nparent != parent != nchild:
                path = path[:-2] + (nchild, )
            cmds["/".join(path)] = {}
            parent, child = nparent, nchild
    # cleanup between loading different tools
    ProxyArgumentParser.reset()
    for k in rm:
        cmds.pop(k, None)
    _COMMANDS.setdefault(tool, {})
    _COMMANDS[tool][cond] = cmds
    return cmds

