# -*- coding: UTF-8 -*-
__all__ = ["Unpacker"]

__initialized = False


def __init():
    global __initialized
    from .__common__ import _init_base
    from ..executable import Executable
    Base = _init_base()
    
    class Unpacker(Base):
        """ Unpacker abstraction.
        
        Extra methods:
          .unpack(executable, **kwargs) [str]
        """
        def unpack(self, executable, **kwargs):
            """ Runs the unpacker according to its command line format and checks if the executable has been changed by
                 this execution. """
            # check: is this unpacker able to process the input executable ?
            exe = Executable(executable)
            if not self._check(exe):
                return
            # now unpack the input executable, taking its hash in order to check for changes
            h = exe.hash
            self._error = None
            label = self.run(exe, extra_opt="-d", **kwargs)
            exe._reset()
            if self._error:
                self.logger("unpacker failed")
                return NOT_LABELLED
            elif h == exe.hash:
                self.logger.debug("not unpacked (content not changed)")
                self._bad = True
                return NOT_LABELLED
            # if unpacking succeeded, we can return packer's label
            self.logger.debug(f"{exe.filename} unpacked using {label}")
            return label
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # dynamically makes Unpacker's registry of child classes from the default dictionary of unpackers
        #  (~/.packing-box/conf/unpackers.yml)
        Unpacker.config = None
    return Unpacker
Unpacker = lazy_object(__init)

