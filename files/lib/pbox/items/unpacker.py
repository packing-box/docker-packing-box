# -*- coding: UTF-8 -*-
from tinyscript import hashlib

from .__common__ import *
from .executable import Executable


# this list is filled in with subclasses at the end of this module
__all__ = ["Unpacker"]


class Unpacker(Base):
    """ Unpacker abstraction. """
    def unpack(self, executable, **kwargs):
        """ Runs the unpacker according to its command line format and checks if the executable has been changed by this
             execution. """
        # check: is this unpacker able to process the input executable ?
        exe = Executable(executable)
        if exe.category not in self._categories_exp or \
           exe.extension[1:] in getattr(self, "exclude", {}).get(exe.category, []):
            return False
        # now unpack the input executable, taking its SHA256 in order to check for changes
        s256 = hashlib.sha256_file(str(exe))
        self._error = False
        label = self.run(exe, **kwargs)
        if self._error:
            return
        elif s256 == hashlib.sha256_file(str(exe)):
            self.logger.debug("%s's content was not changed by %s" % (exe.filename, self.name))
            self._bad = True
            return
        # if unpacking succeeded, we can return packer's label
        self.logger.debug("%s unpacked by %s using %s" % (exe.filename, self.name, label))
        return label


# dynamically makes Unpacker's registry of child classes from the dictionary of unpackers
make_registry(Unpacker)

