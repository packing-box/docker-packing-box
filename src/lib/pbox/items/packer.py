# -*- coding: UTF-8 -*-
from tinyscript import b, ensure_str, hashlib, random, re, subprocess
from tinyscript.helpers import execute_and_log as run

from .__common__ import *
from ..common.executable import Executable
from ..common.item import update_logger
from ..common.utils import make_registry


# this list is filled in with subclasses at the end of this module
__all__ = ["Packer"]


class Packer(Base):
    """ Packer abstraction.
    
    Extra methods:
      .pack(executable, **kwargs) [str]
    
    Overloaded methods:
      .help()
      .run(executable, **kwargs) [str|(str,time)]
    """
    def help(self):
        try:
            return super(Packer, self).help({'families': ",".join(self.families)})
        except AttributeError:
            return super(Packer, self).help()
    
    @update_logger
    def pack(self, executable, include_hash=False, **kwargs):
        """ Runs the packer according to its command line format and checks if the executable has been changed by this
             execution. """
        # check: is this packer able to process the input executable ?
        exe = Executable(executable)
        if not self._check(exe):
            return (None, False) if include_hash else False
        # now pack the input executable, taking its SHA256 in order to check for changes
        h, self._error = exe.hash, None
        label = self.run(exe, **kwargs)
        exe.hash = hashlib.sha256_file(str(exe))
        if self._error:
            err = self._error.replace(str(exe) + ": ", "").replace(self.name + ": ", "").strip()
            self.logger.debug("not packed (%s)" % err)
            return (h, None) if include_hash else None
        elif h == exe.hash:
            self.logger.debug("not packed (content not changed)")
            self._bad = True
            return (h, None) if include_hash else None
        # if packing succeeded, we can return packer's label
        self.logger.debug("packed successfully")
        return (h, label) if include_hash else label
    
    def run(self, executable, **kwargs):
        """ Customizable method for shaping the command line to run the packer on an input executable. """
        # inspect steps and set custom parameters for non-standard packers
        for step in getattr(self, "steps", ["%s %s" % (self.name, executable)]):
            if "{{password}}" in step and 'password' not in self._params:
                self._params['password'] = [random.randstr()]
            m = re.search(r"{{(.*?)\[(.*?)\]}}", step)
            if m and m.group(1) not in self._params:
                self._params[m.group(1)] = [" " + x if x.startswith("-") else x for x in m.group(2).split("|")]
        # then execute parent run() method taking the parameters into account
        return super(Packer, self).run(executable, **kwargs)


# ------------------------------------------------ NON-STANDARD PACKERS ------------------------------------------------
class Ezuri(Packer):
    key = None
    iv  = None
    
    @update_logger
    def run(self, executable, **kwargs):
        """ This packer prompts for parameters. """
        P = subprocess.PIPE
        p = subprocess.Popen(["ezuri"], stdout=P, stderr=P, stdin=P)
        executable = Executable(executable)
        self.logger.debug("ezuri ; inputs: src/dst=%s, procname=%s" % (executable, executable.stem))
        out, err = p.communicate(b("%(e)s\n%(e)s\n%(n)s\n%(k)s\n%(iv)s\n" % {
            'e': executable, 'n': executable.stem,
            'k': "" if Ezuri.key is None else Ezuri.key,
            'iv': "" if Ezuri.iv is None else Ezuri.iv,
        }))
        for l in out.splitlines():
            l = ensure_str(l)
            if not l.startswith("[?] "):
                self.logger.debug(l)
            if Ezuri.key is None and "Random encryption key (used in stub):" in l:
                Ezuri.key = l.split(":", 1)[1].strip()
            if Ezuri.iv is None and "Random encryption IV (used in stub):" in l:
                Ezuri.iv = l.split(":", 1)[1].strip()
        if err:
            self.logger.error(ensure_str(err.strip()))
            self._error = True
        return "%s[key:%s;iv:%s]" % (self.name, Ezuri.key, Ezuri.iv)
# ----------------------------------------------------------------------------------------------------------------------

# dynamically makes Packer's registry of child classes from the dictionary of packers
make_registry(Packer)

