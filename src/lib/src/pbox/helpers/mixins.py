# -*- coding: UTF-8 -*-
__all__ = ["CustomReprMixin", "GetItemMixin", "ResetCachedPropertiesMixin"]

set_exception("KeyNotAllowedError", "KeyError")


class CustomReprMixin:
    """ Sets a custom representation based on the 'name' attribute if it exists. """
    def __repr__(self):
        name = "" if not hasattr(self, "name") else f" ({self.name})"
        return f"<{self.__class__.__name__}{name} object at 0x{id(self):02x}>"


class GetItemMixin:
    """ Allows to call attributes like dictionary keys. """
    def __getitem__(self, name):
        if not isinstance(name, str):
            return super().__getitem__(name)
        if not name.startswith("_"):
            try:
                name, path = name.split(".", 1)
                self[name]  # force patching the target object with the __getitem__ method
                return getattr(self, name).__getitem__(path)
            except ValueError:
                return getattr(self, name)
        raise KeyNotAllowedError(name)


class ResetCachedPropertiesMixin:
    """ Deletes cached_property values to reset cached values. """
    def _reset(self):
        for attr in list(self.__dict__.keys()):
            if not isinstance(getattr(self.__class__, attr, None), cached_property):
                continue
            try:
                delattr(self, attr)
            except AttributeError:
                pass

