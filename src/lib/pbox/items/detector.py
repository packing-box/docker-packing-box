# -*- coding: UTF-8 -*-
from .__common__ import *
from ..common.executable import Executable
from ..common.item import update_logger
from ..common.utils import class_or_instance_method, file_or_folder_or_dataset, make_registry


# this list is filled in with subclasses at the end of this module
__all__ = ["Detector"]


THRESHOLD = lambda l: round(l / 2. + .5)  # majority vote


class Detector(Base):
    """ Detector abstraction.
    
    Extra methods:
      .detect(executable, multiclass, **kwargs) [str]
    
    Overloaded methods:
      .check(*formats, **kwargs)
      .test(executable, multiclass, **kwargs)
    """
    use_output = True
    
    def check(self, *formats, **kwargs):
        """ Checks if the current item is applicable to the given formats. """
        d_mc, i_mc = getattr(self, "multiclass", True), kwargs.pop('multiclass')
        vote, chk_vote = getattr(self, "vote", True), kwargs.pop('vote', True)
        if super(Detector, self).check(*formats, **kwargs):
            # detector can be disabled either because it cannot vote or because it is not multiclass and detection was
            #  requested as multiclass (note that, on the other side, a multiclass-capable detector shall always be
            #  able to output a non-multiclass result (no packer label becomes False, otherwise True)
            if (not chk_vote or vote) and not (not d_mc and i_mc):
                return True
            if chk_vote and not vote:
                self.logger.debug("not allowed to vote")
            if not d_mc and i_mc:
                self.logger.debug("does not support multiclass")
        return False
    
    @class_or_instance_method
    @file_or_folder_or_dataset
    @update_logger
    def detect(self, executable, multiclass=True, **kwargs):
        """ If called from the class:
            Runs every known detector on the given executable and decides the label through voting (with a penalty on
             cases where the executable is considered not packed).
        If called from an instance:
            Runs the detector according to its command line format and checks if the executable has been changed by this
             execution. """
        dslen = kwargs.get('dslen')
        if dslen:
            executable.len = dslen
        l = kwargs.get('label')
        actual_label = [(), (l if multiclass else l is not None, )]['label' in kwargs]
        if isinstance(self, type):
            registry = [d for d in kwargs.get('select', Detector.registry) \
                        if d.check(Executable(executable).format, multiclass=multiclass, **kwargs)]
            l, t = len(registry), kwargs.get('threshold', THRESHOLD) or THRESHOLD
            t = t(l) if isinstance(t, type(lambda: 0)) else t
            if not 0 < t <= l:
                raise ValueError("Bad threshold value, not in [1., %d]" % l)
            results, details = {None: -l + t}, {}
            for detector in registry:
                mc = getattr(detector, "multiclass", True)
                # do not consider Yes|No-detectors if the multiclass option is set
                if multiclass and not mc:
                    continue
                label = list(detector.detect(executable, **kwargs))[0][1]
                if label == -1 and mc:
                    continue
                # if the multiclass option is unset, convert the result to Yes|No
                if not multiclass and mc:
                    label = label is not None
                results.setdefault(label, 0)
                results[label] += 1
                details[detector.name] = label
            # select the best label
            r = (executable, max(results, key=results.get)) + actual_label
            # apply thresholding
            if results[r[1]] < t:
                r = (executable, None) + actual_label
            if kwargs.get("debug", False):
                r += (details, )
            return r
        else:
            e = executable if isinstance(executable, Executable) else Executable(executable)
            if not self.check(e.format, multiclass=multiclass, vote=False, **kwargs) or \
               e.extension[1:] in getattr(self, "exclude", {}).get(e.format, []):
                return -1
            # try to detect a packer on the input executable
            label = self.run(e, **kwargs)
            if kwargs.get('verbose', False):
                print("")
            # if packer detection succeeded, we can return packer's label
            if label:
                label = label.strip()
            else:
                self.logger.debug("did not detect anything")
            if not multiclass:
                label = label is not None if getattr(self, "multiclass", True) else label.lower() == "true"
            if dslen:
                e.len = dslen
            return (e, label) + actual_label
    
    @file_or_folder_or_dataset
    @update_logger
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

