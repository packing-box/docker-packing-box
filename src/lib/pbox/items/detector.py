# -*- coding: UTF-8 -*-
from .__common__ import *
from ..common.executable import Executable
from ..common.utils import class_or_instance_method, file_or_folder_or_dataset, make_registry


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
    def detect(self, executable, multiclass=True, **kwargs):
        """ If called from the class:
            Runs every known detector on the given executable and decides the label through voting (with a penalty on
             cases where the executable is considered not packed).
        If called from an instance:
            Runs the detector according to its command line format and checks if the executable has been changed by this
             execution. """
        l = kwargs.get('label')
        actual_label = [(), (l if multiclass else l is not None, )]['label' in kwargs]
        if isinstance(self, type):
            registry = [d for d in Detector.registry if d.check(Executable(executable).category) and \
                                                        getattr(d, "vote", True)]
            results, details = {None: -len(registry) + kwargs.get('threshold', 3)}, {}
            for detector in registry:
                # do not consider Yes|No-detectors if the multiclass option is set
                if multiclass and not getattr(detector, "multiclass", True):
                    continue
                label = list(detector.detect(executable, **kwargs))[0][1]
                if label == -1 and getattr(detector, "multiclass", True):
                    continue
                # if the multiclass option is unset, convert the result to Yes|No
                if not multiclass and getattr(detector, "multiclass", True):
                    label = label is not None
                results.setdefault(label, 0)
                results[label] += 1
                details[detector.name] = label
            label = max(results, key=results.get)
            r = (executable, label) + actual_label
            if kwargs.get("debug", False):
                r += (details, )
            return r
        else:
            # check: is this detector able to process the input executable ?
            e = executable if isinstance(executable, Executable) else Executable(executable)
            if e.category not in self._categories_exp or \
               e.extension[1:] in getattr(self, "exclude", {}).get(e.category, []):
                return -1
            # now try to detect a packer on the input executable
            label = self.run(e, **kwargs)
            # if packer detection succeeded, we can return packer's label
            if label:
                self.logger.debug("%s detected as packed with %s by %s" % (e.filename, label, self.name))
                label = label.strip()
            if not multiclass:
                label = label is not None if getattr(self, "multiclass", True) else label.lower() == "true"
            return (e, label) + actual_label
    
    @file_or_folder_or_dataset
    def test(self, executable, multiclass=True, **kwargs):
        """ Tests the given item on some executable files. """
        self._test(kwargs.get('silent', False))
        label = self.detect(executable, multiclass, **kwargs)
        real = realv = kwargs.get('label', -1)
        if multiclass:
            if label is None:
                msg = "{} is not packed".format(executable)
            else:
                msg = "{} is packed with {}".format(executable, label)
        else:
            msg = "{} is {}packed".format(executable, ["not ", ""][label])
        if real != -1:
            msg += " ({})".format("not packed" if realv in [None, False] else "packed" if realv is True else realv)
        (self.logger.warning if real == -1 else [self.logger.failure, self.logger.success][label == realv])(msg)


# dynamically makes Detector's registry of child classes from the dictionary of detectors
make_registry(Detector)

