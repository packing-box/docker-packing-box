# -*- coding: UTF-8 -*-
from .__common__ import *
from ..common.executable import Executable
from ..common.item import update_logger


# this list is filled in with subclasses at the end of this module
__all__ = ["Analyzer"]


class Analyzer(Base):
    """ Analyzer abstraction.
    
    Extra methods:
      .analyze(executable, **kwargs) [str]
    """
    use_output = True
    
    @update_logger
    def analyze(self, executable, **kwargs):
        """ Runs the analyzer according to its command line format. """
        e = executable if isinstance(executable, Executable) else Executable(executable)
        if not self.check(e.format, **kwargs) or \
           e.extension[1:] in getattr(self, "exclude", {}).get(e.format, []):
            return -1
        # try to analyze the input executable
        output = self.run(e, **kwargs)
        # if packer detection succeeded, we can return packer's label
        return output


# dynamically makes Analyzer's registry of child classes from the default dictionary of analyzers (~/.opt/analyzers.yml)
Analyzer.source = None

