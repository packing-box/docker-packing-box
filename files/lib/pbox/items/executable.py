# -*- coding: UTF-8 -*-
from functools import cached_property
from magic import from_file
from tinyscript import hashlib
from tinyscript.helpers import is_filetype, Path


__all__ = ["expand_categories", "Executable"]


CATEGORIES = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     ["PE32", "PE64"],
}
SIGNATURES = {
    '^Mach-O 32-bit ':                         "Mach-O32",
    '^Mach-O 64-bit ':                         "Mach-O64",
    '^Mach-O universal binary ':               "Mach-Ou",
    '^MS-DOS executable\s*':                   "MSDOS",
    '^PE32\+? executable (.+?)\.Net assembly': ".NET",
    '^PE32 executable ':                       "PE32",
    '^PE32\+ executable ':                     "PE64",
    '^(set[gu]id )?ELF 32-bit ':               "ELF32",
    '^(set[gu]id )?ELF 64-bit ':               "ELF64",
}


#TODO: move this to learning submodule
FEATURES = {
    'All': {
        'checksum': None,
    },
}


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


class Executable(Path):
    @cached_property
    def category(self):
        best_fmt, l = None, 0
        for ftype, fmt in SIGNATURES.items():
            if len(ftype) > l and is_filetype(str(self), ftype):
                best_fmt, l = fmt, len(ftype)
        return best_fmt
    
    @cached_property
    def features(self):
        return []
    
    @cached_property
    def filetype(self):
        return from_file(str(self))
    
    @cached_property
    def hash(self):
        return hashlib.sha256_file(str(self))


class Features(dict):
    #TODO: move this to learning submodule
    """ This class represents the dictionary of features valid for a given list of executable categories. """
    def __init__(self, *categories):
        categories = expand_categories(*categories)
        all_categories = expand_categories("All")
        # consider most generic features first
        for category, features in FEATURES.items():
            if category in all_categories:
                continue
            for subcategory in expand_categories(category):
                if subcategory in categories:
                    for name, func in features.items():
                        self[name] = func
        # then consider most specific ones
        for category, features in FEATURES.items():
            if category not in all_categories or category not in categories:
                continue
            for name, func in features.items():
                self[name] = func

