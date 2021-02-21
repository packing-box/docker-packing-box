# -*- coding: UTF-8 -*-
from time import time


__all__ = ["benchmark", "compute", "highlight_best", "process"]


bold = lambda text: "\033[1m{}\033[0m".format(text)


def benchmark(f):
    """ Decorator for benchmarking function executions. """
    def _wrapper(*args, **kwargs):
        logger = kwargs.get("logger")
        info = kwargs.pop("info", None)
        start = time()
        r = f(*args, **kwargs)
        info = "" if info is None else "[{}]".format(info)
        message = "{}{}: {} seconds".format(f.__name__, info, time() - start)
        if logger is None:
            print(message)
        else:
            logger.debug(message)
        return r
    return _wrapper


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

