# -*- coding: UTF-8 -*-
from pboxtools.utils import list_tools
from tinyscript import re, sys
from tinyscript.argreparse import ArgumentParser
from tinyscript.helpers import get_parsers, CompositeKeyDict as ckdict, Path, PythonPath
from tinyscript.parser import ProxyArgumentParser


__all__ = ["get_commands"]


_COMMANDS = {}


def get_commands(tool, cond="", category=None, logger=None):
    if tool in _COMMANDS and cond in _COMMANDS[tool]:
        return _COMMANDS[tool][cond]
    if isinstance(tool, str):
        tool = Path(f"~/.opt/tools/{tool}", expand=True)
    parent, child, ref_psr, cmds, rm = None, None, ('main', ), ckdict(_separator_="|"), []
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
                cmds[child][nchild] = ckdict(_separator_="|")
                cmds[child]['_parent'] = cmds
                cmds = cmds[child]
                ref_psr += (child, )
            # depth does not change
            elif nparent == parent or parent is None:
                cmds[nchild] = ckdict(_separator_="|")
            # depth decreases
            elif len(ref_psr) > 1 and nparent == ref_psr[-2]:
                cmds = cmds.pop('_parent', cmds)
                cmds[nchild] = ckdict(_separator_="|")
                ref_psr = ref_psr[:-1]
            # unexpected
            else:
                raise ValueError(f"Unexpected condition while state change ({parent},{child}) => ({nparent},{nchild})")
            parent, child = nparent, nchild
    # cleanup between loading different tools
    ProxyArgumentParser.reset()
    for k in rm:
        cmds.pop(k, None)
    _COMMANDS.setdefault(tool, {})
    _COMMANDS[tool][cond] = cmds
    return cmds

