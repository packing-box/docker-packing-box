# -*- coding: UTF-8 -*-
from tinyscript import hashlib

from .__common__ import *
from ..common.executable import Executable
from ..common.item import update_logger
from ..common.utils import make_registry


# this list is filled in with subclasses at the end of this module
__all__ = ["Unpacker"]


class Unpacker(Base):
    """ Unpacker abstraction.
    
    Extra methods:
      .unpack(executable, **kwargs) [str]
    """
    @update_logger
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
        self._error = None
        label = self.run(exe, **kwargs)
        if self._error:
            return
        elif s256 == hashlib.sha256_file(str(exe)):
            self.logger.debug("%s's content was not changed" % exe.filename)
            self._bad = True
            return
        # if unpacking succeeded, we can return packer's label
        self.logger.debug("%s unpacked using %s" % (exe.filename, label))
        return label


# dynamically makes Unpacker's registry of child classes from the dictionary of unpackers
make_registry(Unpacker)

