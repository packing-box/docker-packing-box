# -*- coding: UTF-8 -*-
from tinyscript.helpers import entropy

from .elf import *
from .pe import *
from ...common.utils import expand_categories, CATEGORIES


__all__ = ["Features", "FEATURE_DESCRIPTIONS"]


FEATURE_DESCRIPTIONS = {}
FEATURE_DESCRIPTIONS.update(PEFEATS)
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
    """ This class represents the dictionary of features valid for a given list of executable categories. """
    def __init__(self, *categories, **kw):
        categories, all_categories = expand_categories(*categories), expand_categories("All")
        select = kw.get('select')
        if select is not None and not isinstance(select, list):
            select = [select]
        # consider most specific features first, then intermediate classes and finally the collapsed class "All"
        l = list(CATEGORIES.keys())
        for cat in [all_categories, l[1:], [l[0]]]:
            for c in cat:
                if any(c2 in expand_categories(c) for c2 in categories):
                    for name, func in FEATURES.get(c, {}).items():
                        self[name] = func

