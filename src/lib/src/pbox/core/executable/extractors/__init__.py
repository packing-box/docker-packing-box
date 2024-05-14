# -*- coding: UTF-8 -*-
from .elf import *
from .macho import *
from .pe import *
from ....helpers.formats import expand_formats


__all__ = ["Extractors", "elfeats", "mofeats", "pefeats"]


_EXTRACTORS = {
    'ELF': {
        'elfeats': elfeats,
    },
    'Mach-O': {
        'mofeats': mofeats,
    },
    'MSDOS': {
        'pefeats': pefeats,
    },
    'PE': {
        'pefeats': pefeats,
    },
}


class Extractors(dict):
    """ This class represents the dictionary of features to be extracted for a given list of executable formats. """
    registry = None
    
    def __init__(self, exe):
        # parse YAML features definition once
        if Extractors.registry is None:
            Extractors.registry = {}
            # consider most specific features first, then intermediate classes and finally the collapsed class "All"
            for flist in [expand_formats("All"), list(FORMATS.keys())[1:], ["All"]]:
                for f in flist:
                    for name, func in _EXTRACTORS.get(f, {}).items():
                        for f2 in expand_formats(f):
                            Extractors.registry.setdefault(f2, {})
                            Extractors.registry[f2].setdefault(name, func)
        if exe is not None:
            for name, func in Extractors.registry.get(exe.format, {}).items():
                self[name] = func(exe)

