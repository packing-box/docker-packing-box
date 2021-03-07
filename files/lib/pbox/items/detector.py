# -*- coding: UTF-8 -*-
from operator import itemgetter

from .__common__ import *
from .executable import Executable


# this list is filled in with subclasses at the end of this module
__all__ = ["Detector"]


# based on: https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod
class class_or_instance_method(classmethod):
    def __get__(self, ins, typ):
        return (super().__get__ if ins is None else self.__func__.__get__)(ins, typ)


class Detector(Base):
    """ Detector abstraction. """
    use_output = True
    
    @class_or_instance_method
    def detect(self, executable, **kwargs):
        """ If called from the class:
            Runs every known detector on the given executable and decides the label through majority voting.
        If called from an instance:
            Runs the detector according to its command line format and checks if the executable has been changed by this
             execution. """
        if isinstance(self, type):
            results = {}
            for detector in Detector.registry:
                label = detector.detect(executable, **kwargs)
                results.setdefault(label, 0)
                results[label] += 1
            return max(results.items(), key=itemgetter(1))[0]
        else:
            # check: is this detector able to process the input executable ?
            e = Executable(executable)
            if e.category not in self._categories_exp or \
               e.extension[1:] in getattr(self, "exclude", {}).get(e.category, []):
                return False
            # now try to detect a packer on the input executable
            label = self.run(e, **kwargs)
            # if packer detection succeeded, we can return packer's label
            if label:
                self.logger.debug("%s detected as packed with %s by %s" % (e.filename, label, self.name))
            return label


# dynamically makes Detector's registry of child classes from the dictionary of detectors
make_registry(Detector)

