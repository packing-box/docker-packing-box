# -*- coding: UTF-8 -*-
from contextlib import contextmanager
from tinyscript import functools, os, subprocess
from tinyscript.helpers import Path, TempPath

from ..core.config import PBOX_HOME


__all__ = ["data_to_temp_file", "edit_file", "figure_path"]


@contextmanager
def data_to_temp_file(data, prefix="temp"):
    """ Save the given pandas.DataFrame to a temporary file. """
    p = TempPath(prefix=prefix, length=8)
    f = p.tempfile("data.csv")
    data.to_csv(str(f), sep=";", index=False, header=True)
    yield f
    p.remove()


def edit_file(path, csv_sep=";", text=False, **kw):
    """" Edit a target file with visidata. """
    cmd = "%s %s" % (os.getenv('EDITOR'), path) if text else "vd %s --csv-delimiter \"%s\"" % (path, csv_sep)
    l = kw.pop('logger', None)
    if l:
        l.debug(cmd)
    subprocess.call(cmd, stderr=subprocess.PIPE, shell=True, **kw)


def figure_path(f):
    """ Decorator for computing the path of a figure given the filename returned by the wrapped function ;
         put it in the "figures" subfolder of the current experiment's folder if relevant. """
    @functools.wraps(f)
    def _wrapper(*a, **kw):
        exp, filename = PBOX_HOME.joinpath("experiment.env"), f(*a, **kw)
        if exp.exists():
            filename = str(Path(exp.read_text()).joinpath("figures", filename))
        return filename
    return _wrapper

