# -*- coding: UTF-8 -*-
import re
from tinyscript.report import *

from ..helpers import *


__all__ = ["References"]

_MAX_COLS_DEFAULT = 12
_MAX_COLS_PER_FIELD = {
    'publisher': 10,
    'year':   20,
}
_PUBLISHERS = {
    'api.semanticscholar.org': "SemanticScholar",
    'arxiv.org': "arXiv",
    'dl.acm.org': "ACM",
    'ieeexplore.ieee.org': "IEEE",
    'link.springer.com': "Springer",
    'www.elsevier.com': "Elsevier",
    'www.mdpi.com': "MDPI",
    'www.mecs-press.org': "MECS Press",
    'www.sciencedirect.com': "ScienceDirect",
}


class References(dict, metaclass=MetaBase):
    _has_registry = False
    
    def __init__(self):
        try:
            self.__initialized
        except AttributeError:
            for name, params in load_yaml_config(self.__class__.config):
                self[name] = params
            self.__initialized = True
    
    def items(self):
        for name, data in super().items():
            d = {'name': name}
            for k, v in data.items():
                d[k] = v
            yield name, d
    
    def values(self):
        for _, data in self.items():
            yield data
    
    @classmethod
    def show(cls, **kw):
        """ Show an overview of the references. """
        cls.logger.debug(f"computing references overview...")
        counts = {'author': {}, 'publisher': {}, 'tag': {}, 'year': {}}
        for data in cls().values():
            for author in data.get('authors'):
                author = author.replace(", ", ",\n")
                counts['author'].setdefault(author, 0)
                counts['author'][author] += 1
            publisher = _PUBLISHERS.get(data.get('url', "").split("://", 1)[-1].split("/", 1)[0], "others")
            counts['publisher'].setdefault(publisher, 0)
            counts['publisher'][publisher] += 1
            for tag in data.get('tags'):
                counts['tag'].setdefault(tag, 0)
                counts['tag'][tag] += 1
            try:
                year = re.search(r"(\d{4})", data.get('date', "")).group()
                counts['year'].setdefault(year, 0)
                counts['year'][year] += 1
            except AttributeError:
                pass
        if len(counts):
            for k, data in counts.items():
                if len(data := sorted(data.items(), key=lambda x: -x[1])):
                    others, data = dict(data).pop('others', 0), [p for p in data if p[0] != "others"]
                    n = _MAX_COLS_PER_FIELD.get(k, _MAX_COLS_DEFAULT)
                    if len(data) > n:
                        data, others = data[:n-1], others + sum(x[1] for x in data[n-1:])
                    h, d = zip(*sorted(data, key=lambda x: x[0].lower()))
                    if others:
                        h += ("others", )
                        d += (others, )
                    render(Section(f"Counts per {k}"), Table([d], column_headers=h))

