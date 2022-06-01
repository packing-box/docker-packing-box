# -*- coding: UTF-8 -*-
import re
from tinyscript.helpers import entropy

from .elf import *
from .pe import *
from .transformers import *
from ...common.utils import expand_formats, FORMATS


__all__ = ["Features", "FEATURE_DESCRIPTIONS", "FEATURE_TRANSFORMERS"]


threshold = lambda x, t, incl=True: x >= t if incl else x > t


FEATURE_DESCRIPTIONS = {}
FEATURE_DESCRIPTIONS.update(ELFEATS)
FEATURE_DESCRIPTIONS.update(PEFEATS)
# expand feature names from regular expressions
FEATURE_TRANSFORMERS = {c: {sk: v for k, v in d.items() for sk in (filter(lambda x: re.search(k, x, re.I),
                            FEATURE_DESCRIPTIONS.keys()) if isinstance(k, str) else [sk])} \
                        for c, d in FEATURE_TRANSFORMERS.items()}
FEATURES = {
    'All': {
        'entropy': lambda exe: entropy(exe.read_bytes()),
    },
    'ELF': {
        'elfeats': lambda exe: elfeats(exe),
    },
    'MSDOS': {
        'pefeats': lambda exe: pefeats(exe),
    },
    'PE': {
        'pefeats': lambda exe: pefeats(exe),
    },
}


class Features(dict):
    """ This class represents the dictionary of features valid for a given list of executable formats. """
    def __init__(self, *formats, **kw):
        formats = expand_formats(*formats)
        # consider most specific features first, then intermediate classes and finally the collapsed class "All"
        l, names = list(FORMATS.keys()), []
        for clist in [expand_formats("All"), l[1:], [l[0]]]:
            for c in clist:
                if any(c2 in expand_formats(c) for c2 in formats):
                    for name, func in FEATURES.get(c, {}).items():
                        self[name] = func

