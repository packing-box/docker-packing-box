# -*- coding: UTF-8 -*-
from contextlib import suppress
from functools import cached_property

from .features import Features
from ..common.executable import Executable as Base


__all__ = ["Executable"]


class Executable(Base):
    """ Executable extension for handling features. """
    def __new__(cls, *parts, **kwargs):
        Features(None)  # lazily populate Features.registry at first instantiation
        self = super(Executable, cls).__new__(cls, *parts, **kwargs)
        if hasattr(self, "_dataset"):
            fields = Executable.FIELDS + ["hash", "label", "Index"]
            with suppress(AttributeError, IndexError): # Attr => 'hash' column missing ; Index => exe does not exist yet
                d = self._dataset._data[self._dataset._data.hash == self.hash].iloc[0].to_dict()
                f = {a: v for a, v in d.items() if a not in fields if str(v) != "nan"}
                if len(f) > 0:
                    setattr(self, "data", f)
        return self
    
    @cached_property
    def data(self):
        return Features(self)
    
    @property
    def features(self):
        if self.format is not None:
            return {n: f.description for n, f in Features.registry[self.format].items()}

