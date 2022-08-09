# -*- coding: UTF-8 -*-
import builtins
from ast import literal_eval

from .common import *
from .elf import *
from .pe import *
from ...common.utils import expand_formats, FORMATS


__all__ = ["Extractors", "Features", "Transformers", "FEATURE_DESCRIPTIONS"]


FEATURES = {
    'All': {
        '256B_block_entropy':             block_entropy(256),
        '256B_block_entropy_per_section': block_entropy_per_section(256),
        '512B_block_entropy':             block_entropy(512),
        '512B_block_entropy_per_section': block_entropy_per_section(512),
        'entropy':                        lambda exe: entropy(exe.read_bytes()),
        'section_characteristics':        section_characteristics,
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

_EVAL_NAMESPACE = {k: getattr(builtins, k) for k in ["abs", "divmod", "float", "hash", "hex", "id", "int", "len",
                                                     "list", "max", "min", "oct", "ord", "pow", "range", "round", "set",
                                                     "str", "sum", "tuple"]}


class Extractors(dict):
    """ This class represents the dictionary of features to be extracted for a given list of executable formats. """
    def __init__(self, *formats, **kw):
        formats = expand_formats(*formats)
        # consider most specific features first, then intermediate classes and finally the collapsed class "All"
        l, names = list(FORMATS.keys()), []
        for clist in [expand_formats("All"), l[1:], [l[0]]]:
            for c in clist:
                if any(c2 in expand_formats(c) for c2 in formats):
                    for name, func in FEATURES.get(c, {}).items():
                        self[name] = func


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
    
    def __call__(self, rawdata):
        _eval = lambda x, d: eval2(x, _EVAL_NAMESPACE, d)
        values = self.get("values", [])
        if len(values) == 0:
            return {self.name: _eval(self.result, rawdata)}
        else:
            d = {}
            for v in values:
                _data = {'x': v}
                _data.update(rawdata)
                n = self.name % v
                d[n] = _eval(self.result, _data)
                FEATURE_DESCRIPTIONS[n] = self.description % v
            return d


class Features(dict):
    def __getitem__(self, name):
        value = super(Features, self).__getitem__(name)
        # if string, this may be a flat list/dictionary converted for working with pandas.DataFrame (cfr error:
        #  ValueError: Must have equal len keys and value when setting with an iterable)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
        return value


class Transformers(dict):
    """ This class parses the YAML definitions of features to be derived from the extracted ones.
        NB: Definitions are not bound to the executable format. """
    def __init__(self, raw_features=None, boolean_only=False):
        for name, params in _FEATURE_TRANSFORMERS.items():
            f = Feature(params, name=name, parent=self)
            if boolean_only and f.boolean or not boolean_only:
                self[name] = f

