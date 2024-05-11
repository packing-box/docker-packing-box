# -*- coding: UTF-8 -*-
from tinyscript import functools, re
from tinyscript.helpers import set_exception


__all__ = ["aggregate_formats", "collapse_formats", "expand_formats", "format_shortname", "get_format_group",
           "ExeFormatDict"]


set_exception("UnknownFormatError", "ValueError")


format_shortname = lambda s, r="": re.sub(r"([-_\.])", r, (s or "").lower())


class ExeFormatDict(dict):
    """ Special dictionary for handling aggregates of sub-dictionaries applying to an executable format, a class of
         formats (depth 1: PE, ELF, Mach-O) or any format (depth 0: All).

    depth 0: All
    depth 1: PE, ELF, Mach-O
    depth 2: PE32, PE64, ELF32, ...
    """
    def __init__(self, *args, **kwargs):
        self.__all = expand_formats("All")
        self.__get = super().__getitem__
        d = args[0] if len(args) > 0 else {}
        d.update(kwargs)
        for i in range(3):
            self.setdefault(i, {})
        for k, v in d.items():
            self[k] = v

    def __delitem__(self, name):
        depth = 0 if name == "All" else 1 if name in FORMATS.keys() else 2 if name in self.__all else -1
        if depth == -1:
            raise UnknownFormatError(f"Unhandled key '{name}'")
        del self.__get(depth)[name]

    def __getitem__(self, name):
        if name not in self.__all:
            raise UnknownFormatError("Bad executable format")
        r = self.__get(0)['All']
        fcls = [k for k in self.__get(1).keys() if name in FORMATS[k]]
        if len(fcls) > 0:
            r.update(self.__get(1)[fcls[0]])
        r.update(self.__get(2).get(name, {}))
        return r

    def __setitem__(self, name, value):
        update = False
        if isinstance(name, (list, tuple)) and len(name) == 2:
            name, update = name
        depth = 0 if name == "All" else 1 if name in FORMATS.keys() else 2 if name in self.__all else -1
        if depth == -1:
            raise UnknownFormatError(f"Unhandled key '{name}'")
        if update:
            self.__get(depth)[name].update(value)
        else:
            self.__get(depth)[name] = value


@functools.lru_cache(maxsize=None)
def aggregate_formats(*formats, **kw):
    """ Aggregate the given input formats. """
    l = []
    for f in formats:
        if isinstance(f, (list, tuple)):
            l.extend(expand_formats(*f))
        else:
            l.append(f)
    return collapse_formats(*set(l)) if kw.get('collapse', False) else list(set(l))


@functools.lru_cache(maxsize=None)
def collapse_formats(*formats, **kw):
    """ 2-depth dictionary-based collapsing function for getting a short list of executable formats. """
    # also support list input argument
    if len(formats) == 1 and isinstance(formats[0], (tuple, list)):
        formats = formats[0]
    selected = [x for x in formats if x is not None]
    # may occur if 'selected' only contains None, meaning that the list of formats provided was built from samples that
    #  are not valid executables
    if len(selected) == 0:
        return []
    groups = [k for k in FORMATS.keys() if k != "All"]
    for f in groups:
        # if a complete group of formats (PE, ELF, Mach-O) is included, only keep the entire group
        if all(x in selected for x in FORMATS[f]):
            for x in FORMATS[f]:
                selected.remove(x)
            selected.append(f)
    # ensure children of complete groups are removed
    for f in selected[:]:
        if f in groups:
            for sf in selected:
                if sf in FORMATS[f]:
                    selected.remove(sf)
    # if everything in the special group 'All' is included, simply select only 'All'
    if all(x in selected for x in FORMATS['All']):
        selected = ["All"]
    return list(set(selected))


@functools.lru_cache(maxsize=None)
def expand_formats(*formats, **kw):
    """ 2-depth dictionary-based expansion function for resolving a list of executable formats. """
    selected = []
    for f in formats:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
        for sf in FORMATS.get(f, [f]):   # depth 2: e.g. ELF => ELF32,ELF64
            if kw.get('once', False):
                selected.append(sf)
            else:
                for ssc in FORMATS.get(sf, [sf]):
                    if ssc not in selected:
                        selected.append(ssc)
    return selected


@functools.lru_cache(maxsize=None)
def get_format_group(exe_format, short=False):
    """ Get the parent formats group from the given executable format. """
    if exe_format is None:
        raise UnknownFormatError("Unknown format")
    for fgroup, formats in list(FORMATS.items())[1:]:  # NB: exclude index 0 as it is "All"
        if exe_format in [fgroup] + formats:
            return format_shortname(fgroup) if short else fgroup
    raise UnknownFormatError(f"Cannot find the group for executable format '{exe_format}'")

