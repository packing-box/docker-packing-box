# -*- coding: UTF-8 -*-
from tinyscript import functools, json, logging
from tinyscript.helpers import is_file, is_folder, lazy_load_module, Path

lazy_load_module("pandas", alias="pd")


__all__ = ["file_or_folder_or_dataset", "filter_data", "filter_data_iter", "get_data", "pd"]

EXTENSIONS = [".json", ".txt"]
__data = None


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


def filter_data(df, query=None, **kw):
    """ Fitler an input Pandas DataFrame based on a given query. """
    i, l = -1, kw.get('logger', logging.nullLogger)
    if query is None or query.lower() == "all":
        return df
    try:
        r = df.query(query)
        if len(r) == 0:
            l.warning("No data selected")
        return r
    except (AttributeError, KeyError) as e:
        l.error("Invalid query syntax ; %s" % e)
    except SyntaxError:
        l.error("Invalid query syntax ; please checkout Pandas' documentation for more information")
    except pd.errors.UndefinedVariableError as e:
        l.error(e)
        l.info("Possible values:\n%s" % "".join("- %s\n" % n for n in df.columns))
    return pd.DataFrame()


def filter_data_iter(df, query=None, limit=0, sample=True, progress=True, transient=False, **kw):
    """ Generator for the filtered data from an input Pandas DataFrame based on a given query. """
    from .rendering import progress_bar
    df = filter_data(df, query, **kw)
    n = len(df.index)
    limit = n if limit <= 0 else min(n, limit)
    if sample and limit < n:
        df = df.sample(n=limit)
    i, genenerator = 0, filter_data(df, query, **kw).itertuples()
    if progress:
        with progress_bar(transient=transient) as p:
            task = p.add_task("", total=limit)
            for row in genenerator:
                yield row
                i += 1
                p.update(task, advance=1.)
                if i >= limit > 0:
                    break
    else:
        for row in genenerator:
            yield row
            i += 1
            if i >= limit > 0:
                break


def get_data(exe_format):
    """ Prepare data for a particular executable format.
    
    Convention for subfolder and file naming: lower() and remove {"-", "_", "."}
      PE     => pe
      Mach-O => macho
    
    Examples:
      Let us take the following structure (NB: ~/.opt/data is the default location defined in config):
        data/
         +-- pe/
         |    +-- common_api_imports.txt
         |    +-- common_api_imports_pe64.txt
         |    +-- common_section_names.txt
         +-- common_section_names.txt
      
     get_data("PE") will output:
      {
        'COMMON_API_IMPORTS':   <list from data/pe/common_api_imports.txt>
        'COMMON_SECTION_NAMES': <list from data/pe/common_section_names.txt>
      }
     
     get_data("ELF") will output:
      {
        'COMMON_SECTION_NAMES': <list from data/common_section_names.txt>
      }
      
     get_data("PE64") will output:
      {
        'COMMON_API_IMPORTS':   <list from data/pe/common_api_imports_pe64.txt>
        'COMMON_SECTION_NAMES': <list from data/pe/common_section_names.txt>
      }
    """
    from .formats import format_shortname as _name, get_format_group
    global __data
    if __data:
        return __data
    _add = lambda a, b: _sort(a.update(b)) if isinstance(a, dict) else list(set(a + b))
    _sort = lambda d: dict(sorted(d.items())) if isinstance(d, dict) else sorted(d or {})
    _uncmt = lambda s: s.split("#", 1)[0].strip()
    # internal file opening function
    def _open(fp):
        # only consider .json or a file formatted (e.g. .txt) with a list of newline-separated strings
        if fp.extension == ".json":
            with fp.open() as f:
                return _sort(json.load(f))
        else:
            return _sort([_uncmt(l) for l in fp.read_text().split("\n") if _uncmt(l) != ""])
    # first, get the group (simply use exe_format if it is precisely a group)
    __data, group = {}, get_format_group(exe_format)
    # consider most specific data first
    if group != exe_format:
        path = config['data'].joinpath(_name(group))
        if path.exists():
            for datafile in path.listdir(lambda p: p.extension in EXTENSIONS):
                if datafile.stem.endswith("_" + _name(exe_format)):
                    __data["_".join(datafile.stem.split("_")[:-1]).upper()] = _open(datafile)
    # then the files without specific mention in a subfolder of config['data'] that matches a format class and
    #  finally the files without specific mention at the root of config['data']
    for path in [config['data'].joinpath(_name(group)), config['data']]:
        if path.exists():
            for datafile in path.listdir(lambda p: p.extension in EXTENSIONS):
                if not datafile.stem.endswith("_" + _name(exe_format)):
                    c = datafile.stem.upper()
                    __data[c] = _add(_open(datafile), __data[c]) if c in __data else _open(datafile)
    return __data

