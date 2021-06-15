# -*- coding: UTF-8 -*-
from .__common__ import *
from .executable import Executable


# this list is filled in with subclasses at the end of this module
__all__ = ["Detector"]


class Detector(Base):
    """ Detector abstraction.
    
    Extra methods:
      .detect(executable, **kwargs) [str]
    """
    use_output = True
    
    @class_or_instance_method
    @file_or_folder_or_dataset
    def detect(self, executable, **kwargs):
        """ If called from the class:
            Runs every known detector on the given executable and decides the label through voting (with a penalty on
             cases where the executable is considered not packed).
        If called from an instance:
            Runs the detector according to its command line format and checks if the executable has been changed by this
             execution. """
        if isinstance(self, type):
            results, details = {None: -len(Detector.registry)}, {}
            for detector in Detector.registry:
                label = detector.detect(executable, **kwargs)
                if label is False:
                    continue
                results.setdefault(label, 0)
                results[label] += 1
                details[detector.name] = label
            label = max(results, key=results.get)
            return (label, details) if kwargs.get("debug", False) else label
        else:
            # check: is this detector able to process the input executable ?
            e = executable if isinstance(executable, Executable) else Executable(executable)
            if e.category not in self._categories_exp or \
               e.extension[1:] in getattr(self, "exclude", {}).get(e.category, []):
                return False
            # now try to detect a packer on the input executable
            label = self.run(e, **kwargs)
            # if packer detection succeeded, we can return packer's label
            if label:
                self.logger.debug("%s detected as packed with %s by %s" % (e.filename, label, self.name))
                label = label.strip()
            return label


# dynamically makes Detector's registry of child classes from the dictionary of detectors
make_registry(Detector)

