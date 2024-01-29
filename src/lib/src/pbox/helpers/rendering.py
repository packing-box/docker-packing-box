# -*- coding: UTF-8 -*-
__all__ = ["progress_bar", "render"]


def progress_bar(unit="samples", silent=False, **kwargs):
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, \
                              TimeRemainingColumn
    class CustomProgress(Progress):
        def track(self, sequence, *args, **kwargs):
            for value in (sequence if silent else super(CustomProgress, self).track(sequence, *args, **kwargs)):
                yield value
    return CustomProgress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn(unit, style="progress.download"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        **kwargs,
    )


def render(*elements, **kw):
    """ Helper function for rendering Tinyscript report objects to the terminal based on a selected backend. """
    from tinyscript.report import Report, Section, Table
    backend = kw.get('backend', DEFAULT_BACKEND)
    if backend == "rich":
        from rich.box import SIMPLE_HEAD
        from rich.console import Console
        from rich.markdown import Heading, Markdown
        from rich.style import Style
        from rich.table import Table as RichTable
        from rich.text import Text as RichText
        from tinyscript import code, colored
        code.replace(Heading.__rich_console__, "text.justify = \"center\"", "")
        _STATUS_CONV = {colored(u, c): RichText(u, style=c) for u, c in zip("☑☑☒☒☐🗗ⓘ☑☒", \
                        ["green", "orange", "red", "magenta", "grey", "cyan", "grey", "grey50", "grey50"])}
        for e in elements:
            if hasattr(e, "md"):
                if isinstance(e, Table):
                    opt = {'show_header': True, 'header_style': "bold", 'highlight': True}
                    if e.title is not None:
                        opt['title'] = e.title
                        opt['title_justify'] = "left"
                        opt['title_style'] = Style(bold=True, color="bright_yellow", italic=False)
                    if getattr(e, "borderless", True):
                        opt['box'] = SIMPLE_HEAD
                    table = RichTable(**opt)
                    for col in e.column_headers:
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
        try:  # from Python3.9
            import mdv3 as mdv
        except ImportError:
            import mdv
        # important notes:
        # - for an unknown reason, mdv breaks tables' layout when used via the packing-box tool and filtering the
        #    category (with -c)
        # - mdv does not support newlines (with <br/>) in the header of a table
        print(mdv.main(elements[0] if len(elements) == 1 and isinstance(elements[0], str) else \
                       Report(*[e for e in elements if e is not None]).md()))
    else:
        raise ValueError("Unknown rendering backend")

