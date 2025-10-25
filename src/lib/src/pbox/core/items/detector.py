# -*- coding: UTF-8 -*-
__all__ = ["Detector"]

__initialized = False


def decide(results, **kwargs):
    """ Decision heuristic of the superdetector.
    
    Voting, if not defined, is based on the majority. It can be set via the 'threshold' option, e.g. "absolute-majority"
     will require a label to get more than the half of all the total votes.
    Note: 'threshold' can be set either as a static integer or as a lambda function depending on the number of voters
    """
    l, t = kwargs['n_detectors'], THRESHOLDS.get(kwargs.get('threshold'), 0)
    t = t(l) if isinstance(t, type(lambda: 0)) else t
    if not 0 <= t <= l:
        raise ValueError(f"Bad threshold value {t:.2f}, not in [0, {l}]")
    vmax = max(results.values())  # best count
    # if the voting heuristic provides a threshold that no label could satisfy, then the superdetector cannot decide
    if vmax < t:
        return NOT_LABELLED
    m = [k for k, v in results.items() if v == vmax]
    # trivial when there is only one maxima
    if len(m) == 1:
        r = m[0]
    # when multiple maxima, only decide if longest match AND shorter strings are include in the longest match ;
    #  otherwise, return NOT_LABELLED (undecided)
    else:
        best = m[0]
        for s in m[1:]:
            if s in best or best in s:
                best = max([s, best], key=len)
            else:
                r = NOT_LABELLED
        r = best
    # apply threshold, keeping NOT_LABELLED if it is the result
    #  (default thresholding is a simple majority, hence threshold == 0 as the majority is handled using the best count)
    return NOT_PACKED if results.get(r, l) < t else r


def __init():
    global __initialized
    from .__common__ import _init_base
    from ..executable import Executable
    from ...helpers import bin_label, class_or_instance_method, file_or_folder_or_dataset
    Base = _init_base()
    
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
            l = self.logger
            i_mc, d_mc = getattr(self, "multiclass", True), kwargs.get('multiclass', True)
            i_vote, d_vote = getattr(self, "vote", True), kwargs.get('vote', False)
            if super(Detector, self).check(*formats, **kwargs):
                # detector can be disabled either because it is not allowed to vote or because it is not multiclass and
                #  detection was requested as multiclass (note that, on the other side, a multiclass-capable detector
                #  shall always be able to output a non-multiclass result (no packer label means False, otherwise True)
                if d_vote and not i_vote:
                    l.debug("not allowed to vote")
                    return False
                if d_mc and not i_mc:
                    warn = not d_vote and not kwargs.get('debug', False)
                    (l.warning if warn else l.debug)(f"{['', self.cname + ' '][warn]}does not support multiclass")
                    return False
                return True
            return False
        
        @class_or_instance_method
        @file_or_folder_or_dataset
        def detect(self, executable, **kwargs):
            """ Detects the packing label(s) of a target executable, folder/dataset of executables, applying the
                 decision heuristic if used as a class (superdetector).
            
            If called from the class:
                Runs every known detector on the given executable and decides the label through voting (with a penalty
                 on cases where the executable is considered not packed).
            
            If called from an instance:
                Runs the detector according to its command line format and outputs its label.
            
            Important note: Detectors are assumed to ouptut
              - binaryclass: NOT_LABELLED, False or True
              - multiclass:  NOT_LABELLED, NOT_PACKED, "unknown" or "[packer-label]"
            """
            label, multiclass, dslen = kwargs.get('label'), kwargs.get('multiclass', True), kwargs.get('dslen')
            kwargs['binary'] = not multiclass
            exe = Executable(executable)
            # in binaryclass, transform the output to -1|0|1
            actual_label = label if multiclass else bin_label(label)
            if dslen:
                exe.len = dslen
            # case (1) called from the class => apply all the in-scope detectors (applicable and with vote=True)
            if isinstance(self, type):
                registry = [d for d in (kwargs.get('select') or Detector.registry) if d.check(exe.format, **kwargs)]
                l = kwargs['n_detectors'] = len(registry)
                results, details = {'unknown': -l} if multiclass else {}, {}
                # step 1: collect strings per packer and suspicions
                kwargs['silent'] = True
                for detector in registry:
                    label = list(detector.detect(exe, **kwargs))[0]  # take the first detection from this detector
                    if isinstance(label, (list, tuple)):
                        label = label[1]  # (Executable, detected_label, expected_label)
                    results.setdefault(label, 0)
                    results[label] += 1
                    details[detector.name] = label
                # step 2: make a decision on the label
                decision = decide(results, **kwargs)
                # format the result, appending details if in debug mode
                r = exe, decision, actual_label
                if kwargs.get("debug", False):
                    r += (details, )
                return r
            # case (2) called from an instance => apply the selected detector if relevant
            else:
                kwargs.pop('vote', None)
                if not self.check(exe.format, vote=False, **kwargs) or \
                   exe.extension[1:] in getattr(self, "exclude", {}).get(exe.format, []):
                    return
                # try to detect a packer on the input executable
                label = self.run(exe, **kwargs)
                if isinstance(label, (list, tuple)):  # when --benchmark is used
                    self.logger.info(f"Computation time: {label[1]:.06f} seconds")
                    label = label[0]
                if kwargs.get('verbose', False):
                    print("")
                # if packer detection succeeded, we can return packer's label
                if label == NOT_LABELLED:
                    self.logger.debug("detector failed")
                elif label == NOT_PACKED:
                    self.logger.debug("did not detect anything")
                else:
                    label = label.strip()
                # if binary classification, convert the result to 1|0|-1 (Yes|No|Not-labelled)
                if not multiclass:
                    label = bin_label(label)
                if dslen:
                    exe.len = dslen
                return exe, label, actual_label
        
        @file_or_folder_or_dataset
        def test(self, executable, **kwargs):
            """ Tests the given item on some executable files. """
            label, multiclass = kwargs.get('label'), kwargs.get('multiclass', True)
            l = self.logger
            self._test(kwargs.get('silent', False))
            try:
                label2 = list(self.detect(executable, **kwargs))[0][1]
            except TypeError:  # NoneType ; when executable has a format not supported by the detector being tested
                if not kwargs.get('verbose', False):
                    l.warning(f"'{executable}' has a format not supported by {self.cname}")
                return
            msg = (f"{executable} is not packed" if label2 == NOT_PACKED else f"{executable} is packed with {label2}") \
                  if multiclass else f"{executable} is {['not ', ''][label2]}packed"
            if label != NOT_LABELLED:
                msg += f" ({'not packed' if label in ['', True] else 'packed' if label is True else label})"
            (l.warning if label == NOT_LABELLED else [l.failure, l.success][label == label2])(msg)
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # dynamically makes Detector's registry of child classes from the default dictionary of detectors
        #  (~/.packing-box/conf/detectors.yml)
        Detector.config = None
    return Detector
Detector = lazy_object(__init)

