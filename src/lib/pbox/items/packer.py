# -*- coding: UTF-8 -*-
from tinyscript import b, ensure_str, hashlib, random, re, subprocess
from tinyscript.helpers import execute_and_log as run

from .__common__ import Base, PARAM_PATTERN
from ..common.executable import Executable
from ..common.item import update_logger
from ..common.utils import make_registry


# this list is filled in with subclasses at the end of this module
__all__ = ["Packer"]


def _parse_parameter(param):
    m = re.match(r"^randint(?:\((\d+),(\d+)\))?$", param)
    if m:
        try:
            i1, i2 = m.groups()
            i1, i2 = i1 or 1, i2 or 256
            return str(random.randint(int(i1), int(i2)))
        except ValueError:
            pass
    return param


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
            return super(Packer, self).help({'categories': ",".join(self.categories)})
        except AttributeError:
            return super(Packer, self).help()
    
    @update_logger
    def pack(self, executable, include_hash=False, **kwargs):
        """ Runs the packer according to its command line format and checks if the executable has been changed by this
             execution. """
        # check: is this packer able to process the input executable ?
        exe = Executable(executable)
        if not self._check(exe):
            return (None, None) if include_hash else None
        # now pack the input executable, taking its SHA256 in order to check for changes
        h, self._error, self._bad = exe.hash, None, False
        label = self.run(exe, **kwargs)
        exe.hash = hashlib.sha256_file(str(exe))
        # if "unmanaged" error, recover from it, without affecting the packer's state ;
        #  Rationale: packer's errors shall be managed beforehand by testing with 'packing-box test packer ...' and its
        #             settings shall be fine-tuned BEFORE making datasets ; "unmanaged" errors should thus not occur
        if self._error:
            err = self._error.replace(str(exe) + ": ", "").replace(self.name + ": ", "").strip()
            self.logger.debug("not packed (%s)" % err)
            return (h, "") if include_hash else ""
        # frequent behaviors ;
        # if packed file's hash was not changed OR a custom failure condition from the packer's definition is met,
        #  then change packer's state to "BAD" ; this will trigger a count at the dataset level and disable the packer
        #  if it triggers too many failures
        elif any(getattr(exe, a, None) == v for a, v in getattr(self, "failure", {}).items()) or h == exe.hash:
            if h == exe.hash:
                self.logger.debug("not packed (content not changed)")
            else:
                for a, v in self.failure.items():
                    if getattr(exe, a, None) == v:
                        self.logger.debug("not packed (failure condition met: %s=%s)" % (a, str(v)))
                        break
            self._bad = True
            return (h, "") if include_hash else ""
        # if packing succeeded, we can return packer's label
        self.logger.debug("packed successfully")
        return (exe.hash, label) if include_hash else label
    
    def run(self, executable, **kwargs):
        """ Customizable method for shaping the command line to run the packer on an input executable. """
        # inspect steps and set custom parameters for non-standard packers
        for step in getattr(self, "steps", ["%s %s" % (self.name, executable)]):
            if "{{password}}" in step and 'password' not in self._params:
                self._params['password'] = [random.randstr()]
            for name, value in re.findall(re.sub(r"\)\?\}\}$", ")}}", PARAM_PATTERN), step):
                if name not in self._params:
                    self._params[name] = [" " + x if x.startswith("-") else _parse_parameter(x) \
                                          for x in value.split("|")]
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
        self.logger.debug("inputs: src/dst=%s, procname=%s" % (executable, executable.stem))
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

