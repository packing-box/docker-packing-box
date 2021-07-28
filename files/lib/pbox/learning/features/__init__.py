# -*- coding: UTF-8 -*-
from tinyscript.helpers import entropy

from .elf import *
from .pe import *
from ...common.utils import expand_categories


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
    def __init__(self, *categories):
        categories, all_categories = expand_categories(*categories), expand_categories("All")
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

