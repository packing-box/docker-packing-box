# -*- coding: UTF-8 -*-
from .__common__ import *
from ..common.executable import Executable
from ..common.item import update_logger
from ..common.utils import class_or_instance_method, file_or_folder_or_dataset


# this list is filled in with subclasses at the end of this module
__all__ = ["Detector"]


THRESHOLD = lambda l: round(l / 2. + .5)  # majority vote


def decide(results, **kwargs):
    """ Decision heuristic of the superdetector. """
    unknown = results.pop("unknown")
    vmax = max(results.values())
    m = [k for k, v in results.items() if v == vmax]
    # trivial when there is only one maxima
    if len(m) == 1:
        return m[0]
    # when multiple maxima, only decide if longest match AND shorter strings are include in the longest match ;
    #  otherwise, return "undecided"
    else:
        best = m[0]
        for s in m[1:]:
            if s in best or best in s:
                best = max([s, best], key=len)
            else:
                return "undecided"
        return best


class Detector(Base):
    """ Detector abstraction.
    
    Extra methods:
      .detect(executable, **kwargs) [str]
    
    Overloaded methods:
      .check(*formats, **kwargs)
      .test(executable, **kwargs)
    """
    use_output = True
    
    def check(self, *formats, **kwargs):
        """ Checks if the current item is applicable to the given formats. """
        d_mc, i_mc = getattr(self, "multiclass", True), kwargs.get('multiclass')
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
    def detect(self, executable, **kwargs):
        """ If called from the class:
            Runs every known detector on the given executable and decides the label through voting (with a penalty on
             cases where the executable is considered not packed).
        If called from an instance:
            Runs the detector according to its command line format. """
        label, multiclass, dslen = kwargs.get('label'), kwargs.get('multiclass', True), kwargs.get('dslen')
        exe = Executable(executable)
        if exe.format is None:  # format is not in the executable SIGNATURES of pbox.common.executable
            self.logger.debug("'%s' is not a valid executable" % exe)
            raise ValueError
        if dslen:
            exe.len = dslen
        actual_label = [(label if multiclass else label != "", ), ()][label is None]
        if isinstance(self, type):
            registry = [d for d in kwargs.get('select', Detector.registry) if d.check(exe.format, **kwargs)]
            l, t = len(registry), kwargs.get('threshold', THRESHOLD) or THRESHOLD
            t = t(l) if isinstance(t, type(lambda: 0)) else t
            if not 0 < t <= l:
                raise ValueError("Bad threshold value, not in [1., %d]" % l)
            results, details = {'unknown': -l + t}, {}
            for detector in registry:
                label = list(detector.detect(exe, **kwargs))[0]
                if isinstance(label, (list, tuple)):
                    label = label[1]
                if label is None:
                    continue
                # if the multiclass option is unset, convert the result to Yes|No
                if not multiclass:
                    label = label != ""
                results.setdefault(label, 0)
                results[label] += 1
                details[detector.name] = label
            # select the best label
            decision = decide(results, **kwargs)
            r = (exe, decision) + actual_label
            # apply thresholding
            if results[r[1]] < t:
                r = (exe, "") + actual_label
            if kwargs.get("debug", False):
                r += (details, )
            return r
        else:
            if not self.check(exe.format, vote=False, **kwargs) or \
               exe.extension[1:] in getattr(self, "exclude", {}).get(exe.format, []):
                return
            # try to detect a packer on the input executable
            label = self.run(exe, **kwargs)
            if kwargs.get('verbose', False):
                print("")
            # if packer detection succeeded, we can return packer's label
            if label:
                label = label.strip()
            else:
                self.logger.debug("did not detect anything")
            if not multiclass:
                label = label != "" if getattr(self, "multiclass", True) else label.lower() == "true"
            if dslen:
                exe.len = dslen
            return (exe, label) + actual_label
    
    @file_or_folder_or_dataset
    @update_logger
    def test(self, executable, **kwargs):
        """ Tests the given item on some executable files. """
        label, multiclass = kwargs.get('label'), kwargs.get('multiclass', True)
        self._test(kwargs.get('silent', False))
        label2 = self.detect(executable, **kwargs)
        if multiclass:
            if label2 == "":
                msg = "{} is not packed".format(executable)
            else:
                msg = "{} is packed with {}".format(executable, label2)
        else:
            msg = "{} is {}packed".format(executable, ["not ", ""][label2])
        if label is not None:
            msg += " ({})".format("not packed" if label in ["", True] else "packed" if label is True else label)
        (self.logger.warning if label is None else [self.logger.failure, self.logger.success][label == label2])(msg)


# dynamically makes Detector's registry of child classes from the default dictionary of detectors (~/.opt/detectors.yml)
Detector.source = None

