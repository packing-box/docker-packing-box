# -*- coding: UTF-8 -*-
import re
import yaml
from tinyscript.helpers import entropy

from .elf import *
from .pe import *
from ...common.utils import expand_formats, FORMATS


__all__ = ["Features", "Transformers", "FEATURE_DESCRIPTIONS"]


_FEATURE_OPERATIONS = {
    'is_0':       lambda x: x == 0,
    'is_eq':      lambda x, v: x == v,
    'is_in':      lambda x, *v: x in v,
    'is_gt':      lambda x, f: x > f,
    'is_gte':     lambda x, f: x >= f,
    'is_lt':      lambda x, f: x < f,
    'is_lte':     lambda x, f: x <= f,
    'is_mult':    lambda x, i: x % i == 0,
    'is_not_in':  lambda x, *v: x not in v,
    'is_not_0':   lambda x: x != 0,
    'is_within':  lambda x, v1, v2: v1 <= x <= v2,
    'multiplier': lambda x, m: x / m if x % m == 0 else -1,
    'threshold':  lambda x, t, incl=True: x >= t if incl else x > t,
}
_FEATURE_OPERATIONS['is_equal'] = _FEATURE_OPERATIONS['is_eq']
__F = {
    'name':   r"[a-z]+(?:_[a-z0-9]+)*",
    'range':  r"(?:\,\s*range\(\d+\,\s*\d+\)(?:\s*\*\s*\d+)*)*",
    'values': r"(?:\,\s*(?:\d+|0x\d+|\[\d+(?:\,\s*\d+)*\]))*",
}
_FEATURE_FORMATS = [
    r"(%s)\(%s%s\)" % ("|".join(_FEATURE_OPERATIONS.keys()), __F['name'], __F['values']),  # classical expression
    r"%(name)s\s*\/\s*%(name)s\s*(?:<|>|<=|>=|==)\s*\d*\.\d+" % {'name': __F['name']},     # ratio
    r"(%s)\(%s%s%s%s\)" % ("|".join(_FEATURE_OPERATIONS.keys()),
                           __F['name'], __F['values'], __F['range'], __F['values']),       # expression with range
]
with open("/opt/features.yml") as f:
    _FEATURE_TRANSFORMERS = yaml.load(f, Loader=yaml.Loader)

FEATURES = {
    'All': {
        'entropy': lambda exe: entropy(exe.read_bytes()),
    },
    'ELF': {
        'elfeats': lambda exe: elfeats(exe),
    },
    #'Mach-O': {  #TODO
    #    '??': lambda exe: ??(exe),
    #},
    'MSDOS': {
        'pefeats': lambda exe: pefeats(exe),
    },
    'PE': {
        'pefeats': lambda exe: pefeats(exe),
    },
}
FEATURE_DESCRIPTIONS = {}
FEATURE_DESCRIPTIONS.update(ELFEATS)
FEATURE_DESCRIPTIONS.update(PEFEATS)


class Feature(dict):
    def __init__(self, idict, **kwargs):
        self.setdefault("name", "undefined")
        self.setdefault("description", "")
        self.setdefault("result", None)
        self.setdefault("results", None)
        super(Feature, self).__init__(idict, **kwargs)
        self['boolean'] = any(self['name'].startswith(p) for p in ["is_", "has_"])
        self.__dict__ = self
        if self.result is None and self.results is None:
            raise ValueError("%s: 'result' or 'results' shall be defined" % self.name)
        elif self.result is not None and self.results is not None:
            raise ValueError("%s: 'result' and 'results' shall not be defined at the same time" % self.name)
        if self.result and all(re.match(r, self.result) is None for r in _FEATURE_FORMATS) or \
           self.results and all(re.match(r, self.results) is None for r in _FEATURE_FORMATS):
            raise ValueError("%s: bad feature expression" % self.name)
    
    def __call__(self, rawdata):
        _eval = lambda x: eval(x, _FEATURE_OPERATIONS, rawdata)
        if self.result:
            return {self.name: _eval(self.result)}
        else:
            d, m = {}, re.search(r"(range\(\d+\,\s*\d+\))", self.results)
            if m:
                rng = m.group(0)
                for i in eval(rng):
                    try:
                        n = self.name % i
                        d[n] = _eval(self.results.replace(rng, str(i)))
                        FEATURE_DESCRIPTIONS[n] = self.description % i
                    except TypeError:
                        raise ValueError("%s: bad feature attribute(s) (%%d missing)" % self.name)
            return d


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


class Transformers(dict):
    """ This class parses the YAML definitions of features. It is not format-bound. """
    def __init__(self, raw_features=None, boolean_only=False):
        for name, params in _FEATURE_TRANSFORMERS.items():
            f = Feature(params, name=name, parent=self)
            if boolean_only and f.boolean or not boolean_only:
                self[name] = f

