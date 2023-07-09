# -*- coding: UTF-8 -*-
from ast import literal_eval
from tinyscript.helpers import execute_and_log as run


__all__ = ["pefeats"]


def pefeats(executable):
    """ This uses pefeats to extract 119 features from PE files. """
    out, err, retc = run("pefeats \'%s\'" % executable)
    if retc == 0:
        values = []
        for x in out.decode().strip().split(",")[1:]:
            try:
                values.append(literal_eval(x))
            except ValueError:
                values.append(x)
        return values

