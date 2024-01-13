# -*- coding: UTF-8 -*-
from contextlib import contextmanager
from tinyscript import functools, os, re, subprocess
from tinyscript.helpers import is_file, is_folder, is_hash, set_exception, Path, TempPath

from .utils import pd


__all__ = ["data_to_temp_file", "edit_file", "file_or_folder_or_dataset", "find_files_in_folder", "Locator", "Path"]


set_exception("BadSchemeError", "ValueError")
set_exception("NotAnExperimentError", "ValueError")


class Locator(Path):
    """ Extension of the Path class from Tinyscript to support special schemes dedicated to Packing Box. """
    def __new__(cls, *parts, **kwargs):
        try:
            scheme, path = str(parts[0]).split("://")
        except ValueError:
            return super(Path, cls).__new__(cls, *parts, **kwargs)
        if scheme in ["conf", "data"]:
            if scheme == "conf" and not path.endswith(".yml"):
                path += ".yml"
            p = config['workspace'].joinpath(scheme, path)
        elif scheme in ["dataset", "model"]:
            from ..core import Dataset, Model
            return locals()[scheme.capitalize()].load(path)
        elif scheme == "executable":
            from ..core import is_exe, Dataset, Executable
            if is_hash(path):
                for ds in Dataset.iteritems():
                    p = ds.joinpath("files", path)
                    if p.exists():
                        return Executable(str(p))
            for srcs in PACKING_BOX_SOURCES.values():
                for src in srcs:
                    for exe in Path(src, expand=True).walk(filter_func=is_exe, sort=False):
                        if exe.basename == path:
                            return Executable(str(exe))
            raise FileNotFoundError(f"[Errno 2] No such executable: '{path}'")
        elif scheme == "experiment":
            from ..core import Experiment
            try:
                return Experiment(config['experiment'])
            except KeyError:
                return Experiment(path)
        elif scheme in ["figure", "script"]:
            try:
                p = config['experiment'].joinpath(scheme + "s", path)
            except KeyError:
                raise NotAnExperimentError("This scheme can only be used in the context of an experiment")
        else:
            raise BadSchemeError(f"'{scheme}' is not a valid scheme")
        if p.exists():
            return p
        raise FileNotFoundError(f"[Errno 2] No such {scheme}: '{path}'")


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


def file_or_folder_or_dataset(method):
    """ This decorator allows to handle, as the first positional argument of an instance method, either an executable,
         a folder with executables or the executable files from a Dataset. """
    @functools.wraps(method)
    def _wrapper(self, *args, **kwargs):
        from ..core.executable import Executable
        kwargs['silent'] = kwargs.get('silent', False)
        # collect executables and folders from args
        n, e, l = -1, [], {}
        # exe list extension function
        def _extend_e(i):
            nonlocal n, e, l
            # append the (Fileless)Dataset instance itself
            if not isinstance(i, Executable) and getattr(i, "is_valid", lambda: False)():
                if not kwargs['silent']:
                    self.logger.debug("input is a (Fileless)Dataset structure")
                for exe in i:
                    e.append(exe)
                return True
            # single executable
            elif is_file(i):
                if not kwargs['silent']:
                    self.logger.debug("input is a single executable")
                if i not in e:
                    i = Path(i)
                    i.dataset = None
                    e.append(i)
                    lbl = kwargs.get('label')
                    if lbl:
                        l = {i.stem: lbl}
                return True
            # normal folder or FilelessDataset's path or Dataset's files path
            elif is_folder(i):
                if not kwargs['silent']:
                    self.logger.debug("input is a folder of executables")
                for f in Path(i).walk(filter_func=lambda p: p.is_file()):
                    f.dataset = None
                    if str(f) not in e:
                        e.append(f)
                return True
            else:
                i = config['datasets'].joinpath(i)
                # check if it has the structure of a dataset
                if all(i.joinpath(f).is_file() for f in ["data.csv", "metadata.json"]):
                    if i.joinpath("files").is_dir() and not i.joinpath("features.json").exists():
                       
                        if not kwargs['silent']:
                            self.logger.debug("input is Dataset from %s" % config['datasets'])
                        data = pd.read_csv(str(i.joinpath("data.csv")), sep=";")
                        l = {exe.hash: exe.label for exe in data.itertuples()}
                        dataset = i.basename
                        for f in i.joinpath("files").listdir():
                            f.dataset = dataset
                            if str(f) not in e:
                                e.append(f)
                        return True
                    # otherwise, it is a FilelessDataset and it won't work as this decorator requires samples
                    self.logger.warning("FilelessDataset is not supported as it does not hold samples to iterate")
            return False
        # use the extension function to parse:
        # - positional arguments up to the last valid file/folder
        # - then the 'file' keyword-argument
        for n, a in enumerate(args):
            # if not a valid file, folder or dataset, stop as it is another type of argument
            if not _extend_e(a):
                break
        args = tuple(args[n+1:])
        for a in kwargs.pop('file', []):
            _extend_e(a)
        # then handle the list
        i, kwargs['silent'] = -1, kwargs.get('silent', False)
        for i, exe in enumerate(e):
            exe = Executable(exe)
            if exe.format is None:  # format is not in the executable SIGNATURES of pbox.core.executable
                self.logger.debug("'%s' is not a valid executable" % exe)
                continue
            kwargs['dslen'] = len(e)
            # this is useful for a decorated method that handles the difference between the computed and actual labels
            kwargs['label'] = l.get(Path(exe).stem, NOT_LABELLED)
            try:
                yield method(self, exe, *args, **kwargs)
            except ValueError as err:
                self.logger.exception(err)
            kwargs['silent'] = True
        if i == -1:
            self.logger.error("No (valid) executable selected")
    return _wrapper


def find_files_in_folder(path):
    """ Utility function to find files based on a path whose basename can be a regex pattern. """
    p = Path(path)
    regex = re.compile(p.basename)
    for fp in Path(p.dirname).listdir(filter_func=lambda x: x.is_file() and regex.search(x.basename)):
        yield fp

