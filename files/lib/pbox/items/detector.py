# -*- coding: UTF-8 -*-
from tinyscript import hashlib

from .__common__ import *
from .executable import Executable


# this list is filled in with subclasses at the end of this module
__all__ = ["Detector"]


class Detector(Base):
    """ Detector abstraction. """
    def detect(self, executable, **kwargs):
        """ Runs the detector according to its command line format and checks if the executable has been changed by this
             execution. """
        # check: is this detector able to process the input executable ?
        exe = Executable(executable)
        if exe.category not in self._categories_exp or \
           exe.extension[1:] in getattr(self, "exclude", {}).get(exe.category, []):
            return False
        # now try to detect a packer on the input executable
        label = self.run(exe, **kwargs)
        # if packer detection succeeded, we can return packer's label
        if label:
            self.logger.debug("%s detected as packed with %s by %s" % (exe.filename, label, self.name))
        return label


# dynamically makes Detector's registry of child classes from the dictionary of detectors
make_registry(Detector)

