# -*- coding: UTF-8 -*-
from rich import box
from rich.console import Console
from rich.markdown import Heading, Markdown
from rich.style import Style
from rich.table import Table as RichTable
from rich.text import Text as RichText
from tinyscript import *
from tinyscript.report import *
try:  # from Python3.9
    import mdv3 as mdv
except ImportError:
    import mdv


__all__ = ["render", "NOK", "NOK_GREY", "OK", "OK_GREY", "STATUS"]

DEFAULT_BACKEND = "rich"
NOK, NOK_GREY = colored("☒", "red"), colored("☒", "grey")
OK, OK_GREY = colored("☑", "green"), colored("☑", "grey")
STATUS = {
    'broken':        colored("☒", "magenta"),
    'commercial':    "💰",
    'gui':           colored("🗗", "cyan"),
    'info':          colored("ⓘ", "grey"),
    'installed':     colored("☑", "orange"),
    'not installed': colored("☒", "red"),
    'ok':            colored("☑", "green"),
    'todo':          colored("☐", "grey"),
    'useless':       colored("ⓘ", "grey"),
}
_STATUS_CONV = {colored(u, c): RichText(u, style=c) for u, c in \
                zip("☒🗗ⓘ☑☒☑☐ⓘ", ["magenta", "cyan", "grey", "orange", "red", "green", "grey", "grey"])}


code.replace(Heading.__rich_console__, "text.justify = \"center\"", "")


def render(*elements, **kw):
    """ Helper function for rendering Tinyscript report objects to the terminal based on a selected backend. """
    backend = kw.get('backend', DEFAULT_BACKEND)
    if backend == "rich":
        for e in elements:
            if hasattr(e, "md"):
                if isinstance(e, Table):
                    opt = {'show_header': True, 'header_style': "bold", 'highlight': True}
                    if e.title is not None:
                        opt['title'] = e.title
                        opt['title_justify'] = "left"
                        opt['title_style'] = Style(bold=True, color="bright_yellow", italic=False)
                    if getattr(e, "borderless", True):
                        opt['box'] = box.SIMPLE_HEAD
                    table = RichTable(**opt)
                    for i, col in enumerate(e.column_headers):
                        table.add_column(col, justify="center")
                    for row in e.data:
                        if not all(set(str(cell).strip()) == {"-"} for cell in row):
                            table.add_row(*[RichText(str(cell), justify="left") if cell not in _STATUS_CONV else \
                                            _STATUS_CONV[cell] for cell in row])
                    Console(markup=False).print(table)
                elif isinstance(e, Section):
                    Console().print(Markdown(e.md()), style=Style(bold=True, color="bright_cyan", italic=False))
                else:
                    Console().print(Markdown(e.md()))
            elif e is not None:
                Console().print(Markdown(e))
    elif backend == "mdv":
        # important notes:
        # - for an unknown reason, mdv breaks tables' layout when used via the packing-box tool and filtering the
        #    category (with -c)
        # - mdv does not support newlines (with <br/>) in the header of a table
        print(mdv.main(elements[0] if len(elements) == 1 and isinstance(elements[0], str) else \
                       Report(*[e for e in elements if e is not None]).md()))
    else:
        raise ValueError("Unknown rendering backend")

