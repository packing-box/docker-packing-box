# -*- coding: UTF-8 -*-
from tinyscript import logging

from tinyscript.report import *


__all__ = ["Item"]


_fmt_name = lambda x: (x or "").lower().replace("_", "-")


class Item:
    """ Item abstraction. """
    def __init__(self):
        self.cname = self.__class__.__name__
        self.name = _fmt_name(self.__class__.__name__)
        self.type = self.__class__.__base__.__name__.lower()
        self.logger = logging.getLogger(self.name)
    
    def __repr__(self):
        """ Custom string representation for an item. """
        return "<%s %s at 0x%x>" % (self.__class__.__name__, self.type, id(self))
    
    def help(self):
        """ Returns a help message in Markdown format. """
        md = Report()
        if getattr(self, "description", None):
            md.append(Text(self.description))
        if getattr(self, "comment", None):
            md.append(Blockquote("**Note**: " + self.comment))
        if getattr(self, "link", None):
            md.append(Blockquote("**Link**: " + self.link))
        if getattr(self, "references", None):
            md.append(Section("References"), List(*self.references, **{'ordered': True}))
        return md.md()
    
    @classmethod
    def get(cls, item, error=True):
        """ Simple class method for returning the class of an item based on its name (case-insensitive). """
        for i in cls.registry:
            if i.name == (item.name if isinstance(item, Item) else _fmt_name(item)):
                return i
        if error:
            raise ValueError("'%s' is not defined" % item)
    
    @classmethod
    def iteritems(cls):
        """ Class-level iterator for returning enabled items. """
        for i in cls.registry:
            try:
                if i.status in i.__class__._enabled:
                    yield i
            except AttributeError:
                yield i

