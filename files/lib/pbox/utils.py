# -*- coding: UTF-8 -*-
from time import time


__all__ = ["benchmark", "expand_categories", "highlight_best"]


CATEGORIES = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     [".NET", "PE32", "PE64"],
}


bold = lambda text: "\033[1m{}\033[0m".format(text)


def benchmark(f):
    """ Decorator for benchmarking function executions. """
    def _wrapper(*args, **kwargs):
        logger = kwargs.get("logger")
        info = kwargs.pop("info", None)
        perf = kwargs.pop("perf", True)
        t = perf_counter if perf else time
        start = t()
        r = f(*args, **kwargs)
        dt = t() - start
        message = "{}{}: {} seconds".format(f.__name__, "" if info is None else "[{}]".format(info), dt)
        if logger is None:
            print(message)
        else:
            logger.debug(message)
        return r
    return _wrapper


def expand_categories(*categories, **kw):
    """ 2-depth dictionary-based expansion function for resolving a list of executable categories. """
    selected = []
    for c in categories:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
        for sc in CATEGORIES.get(c, [c]):   # depth 2: e.g. ELF => ELF32,ELF64
            if kw.get('once', False):
                selected.append(sc)
            else:
                for ssc in CATEGORIES.get(sc, [sc]):
                    if ssc not in selected:
                        selected.append(ssc)
    return selected


def highlight_best(table_data, from_col=2):
    """ Highlight the highest values in the given table. """
    new_data = [table_data[0]]
    maxs = len(new_data[0][from_col:]) * [0]
    # search for best values
    for data in table_data[1:]:
        for i, value in enumerate(data[from_col:]):
            maxs[i] = max(maxs[i], float(value))
    # reformat the table, setting bold text for best values
    for data in table_data[1:]:
        new_row = data[:from_col]
        for i, value in enumerate(data[from_col:]):
            new_row.append(bold(value) if float(value) == maxs[i] else value)
        new_data.append(new_row)
    return new_data

