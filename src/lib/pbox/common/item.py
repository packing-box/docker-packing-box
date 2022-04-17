# -*- coding: UTF-8 -*-
from tinyscript import functools, logging

from tinyscript.report import *


__all__ = ["update_logger", "Item"]


_fmt_name = lambda x: (x or "").lower().replace("_", "-")


def update_logger(m):
    """ Method decorator for triggering the setting of the bound logger (see pbox.common.Item.__getattribute__). """
    @functools.wraps(m)
    def _wrapper(self, *a, **kw):
        getattr(self, "logger", None)
        return m(self, *a, **kw)
    return _wrapper


class Item:
    """ Item abstraction. """
    def __init__(self):
        self.cname = self.__class__.__name__
        self.name = _fmt_name(self.__class__.__name__)
        self.type = self.__class__.__base__.__name__.lower()
        self.logger # triggers the creation of a logger with the default config
    
    def __getattribute__(self, name):
        """ Custom getattribute method for setting the logger. """
        if name == "logger":
            if not hasattr(self, "_logger"):
                self._logger = logging.getLogger(self.name)
                self._logger_init = False
            elif not self._logger_init:
                # this is required for aligning item's log config to the main logger configured with the 'initialize'
                #  function of Tinyscript
                logging.setLogger(self.name)
                self._logger_init = True
            return self._logger
        return super(Item, self).__getattribute__(name)
    
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

