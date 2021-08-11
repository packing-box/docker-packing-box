# -*- coding: UTF-8 -*-
from tinyscript import logging

from tinyscript.report import *


__all__ = ["Item"]


class Item:
    """ Item abstraction. """
    def __init__(self):
        self.cname = self.__class__.__name__
        self.name = self.__class__.__name__.lower().replace("_", "-")
        self.type = self.__class__.__base__.__name__.lower()
        self.logger = logging.getLogger(self.name)
    
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
    def get(cls, item):
        """ Simple class method for returning the class of an item based on its name (case-insensitive). """
        for i in cls.registry:
            if i.name == (item.name if isinstance(item, Item) else item).lower():
                return i

