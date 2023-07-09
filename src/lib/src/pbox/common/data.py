# -*- coding: UTF-8 -*-
from tinyscript import functools, json

from .config import config
from .utils import format_shortname as _name, get_format_group


__all__ = ["get_data"]


EXTENSIONS = [".json", ".txt"]


@functools.lru_cache
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
    _add = lambda a, b: _sort(a.update(b) if isinstance(d, dict) else a + b)
    _const = lambda s: "_".join(s.split("_")[:-1]).upper()
    _sort = lambda d: dict(sorted(d.items())) if isinstance(d, dict) else sorted(d)
    _uncmt = lambda s: s.split("#", 1)[0].strip()
    # internal file opening function
    def _open(fp):
        # only consider .json or a file formatted (e.g. .txt) with a list of newline-separated strings
        with fp.open() as f:
            return _sort(json.load(f) if fp.extension == ".json" else
                         [_uncmt(l) for l in f.read_text().split("\n") if _uncmt(l) != ""])
    # first, get the group (simply use exe_format if it is precisely a group)
    d, group = {}, get_format_group(exe_format)
    # consider most specific data first
    if group != exe_format:
        path = config['data'].joinpath(_name(group))
        if path.exists():
            for datafile in path.listdir(lambda p: p.extension in EXTENSIONS):
                if datafile.filename.endswith("_" + _name(exe_format)):
                    d[_const(datafile.filename)] = _open(datafile)
    # then the files without specific mention in a subfolder of config['data'] that matches a format class and
    #  finally the files without specific mention at the root of config['data']
    for grp, path in [(group, config['data'].joinpath(_name(group))), ("All", config['data'])]:
        path = config['data'].joinpath(_name(group))
        if path.exists():
            for datafile in path.listdir(lambda p: p.extension in EXTENSIONS):
                if not datafile.filename.endswith("_" + _name(exe_format)):
                    c = _const(datafile.filename)
                    d[c] = _add(d[c], _open(datafile)) if c in d else _open(datafile)
    return d

