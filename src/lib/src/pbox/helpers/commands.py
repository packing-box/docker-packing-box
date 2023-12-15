# -*- coding: UTF-8 -*-
from tinyscript.argreparse import ArgumentParser
from tinyscript.helpers import CompositeKeyDict as ckdict, Path, PythonPath
from tinyscript.parser import ProxyArgumentParser


__all__ = ["get_commands"]


def get_commands(tool, cond="", category=None):
    if isinstance(tool, str):
        tool = Path(f"~/.opt/tools/{tool}", expand=True)
    cmds, module = ckdict(_separator_="|"), f"_{tool.basename}.py"
    # copy the target tool to modify it so that its parser tree can be retrieved
    ntool = tool.copy(module)
    ntool.write_text(ntool.read_text().replace("if __name__ == '__main__':", f"{cond}\ndef main():") \
                                      .replace("if __name__ == \"__main__\":", "def main():") \
                                      .replace("initialize(", "return parser\n    initialize(") \
                                      .rstrip("\n") + "\n\nif __name__ == '__main__':\n    main()\n")
    ntool.chmod(0o755)
    # populate the real parser and add information arguments
    try:
        __parsers = {PythonPath(module).module.main(): ArgumentParser(globals())}
    except Exception as e:
        logger.critical(f"Completion generation failed for tool: {tool.basename}")
        logger.error(f"Source ({module}):\n{ntool.read_text()}")
        logger.exception(e)
        sys.exit(1)
    # now import the populated list of parser calls from within the tinyscript.parser module
    from tinyscript.parser import parser_calls, ProxyArgumentParser
    global parser_calls
    #  proxy parser to real parser recursive conversion function
    def __proxy_to_real_parser(value):
        """ Source: tinyscript.parser """
        if isinstance(value, ProxyArgumentParser):
            return __parsers[value]
        elif isinstance(value, (list, tuple)):
            return [__proxy_to_real_parser(_) for _ in value]
        return value
    #  now iterate over the registered calls
    pairs = []
    for proxy_parser, method, args, kwargs, proxy_subparser in parser_calls:
        kw_category = kwargs.get('category')
        real_parser = __parsers[proxy_parser]
        args = (__proxy_to_real_parser(v) for v in args)
        kwargs = {k: __proxy_to_real_parser(v) for k, v in kwargs.items()}
        # NB: when initializing a subparser, 'category' kwarg gets popped
        real_subparser = getattr(real_parser, method)(*args, **kwargs)
        if real_subparser is not None:
            __parsers[proxy_subparser] = real_subparser
        if not isinstance(real_subparser, str):
            real_subparser._parent = real_parser
            real_subparser.category = kw_category  # reattach category
    parent, child, ref_psr, ref_dct, rm = None, None, ('main', ), cmds, []
    for parser in __parsers.values():
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
                ref_dct[child][nchild] = ckdict(_separator_="|")
                ref_dct[child]['_parent'] = ref_dct
                ref_dct = ref_dct[child]
                ref_psr += (child, )
            # depth does not change
            elif nparent == parent or parent is None:
                ref_dct[nchild] = ckdict(_separator_="|")
            # depth decreases
            elif len(ref_psr) > 1 and nparent == ref_psr[-2]:
                ref_dct = ref_dct.pop('_parent', cmds)
                ref_dct[nchild] = ckdict(_separator_="|")
                ref_psr = ref_psr[:-1]
            # unexpected
            else:
                raise ValueError(f"Unexpected condition while state change ({parent},{child}) => ({nparent},{nchild})")
            parent, child = nparent, nchild
    # cleanup between loading different tools
    ProxyArgumentParser.reset()
    ntool.remove()
    for k in rm:
        cmds.pop(k, None)
    return cmds

