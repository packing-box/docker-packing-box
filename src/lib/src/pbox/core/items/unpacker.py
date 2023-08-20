# -*- coding: UTF-8 -*-
from tinyscript.helpers import lazy_object


__all__ = ["Unpacker"]

__initialized = False


def __init():
    global __initialized
    from .__common__ import _init_base
    from ..executable import Executable
    from ...helpers import update_logger
    Base = _init_base()
    
    class Unpacker(Base):
        """ Unpacker abstraction.
        
        Extra methods:
          .unpack(executable, **kwargs) [str]
        """
        @update_logger
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
            exe.update()
            if self._error:
                self.logger("unpacker failed")
                return NOT_LABELLED
            elif h == exe.hash:
                self.logger.debug("not unpacked (content not changed)")
                self._bad = True
                return NOT_LABELLED
            # if unpacking succeeded, we can return packer's label
            self.logger.debug("%s unpacked using %s" % (exe.filename, label))
            return label
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # dynamically makes Unpacker's registry of child classes from the default dictionary of unpackers
        #  (~/.opt/unpackers.yml)
        Unpacker.source = None
    return Unpacker
Unpacker = lazy_object(__init)

