# -*- coding: UTF-8 -*-
from tinyscript import ensure_str, random, re, subprocess


__all__ = ["Packer"]

__initialized = False


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


def __init():
    global __initialized
    from .__common__ import _init_base
    from ..executable import Executable
    Base = _init_base()
    
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
        
        def pack(self, executable, **kwargs):
            """ Runs the packer according to its command line format and checks if the executable has been changed by
                 this execution. """
            # check: is this packer able to process the input executable ?
            exe = Executable(executable)
            if not self._check(exe):
                return
            # now pack the input executable, taking its SHA256 in order to check for changes
            h, fmt, self._error, self._bad = exe.hash, exe.format, None, False
            label = self.run(exe, **kwargs)
            exe._reset()
            # if "unmanaged" error, recover from it, without affecting the packer's state ;
            #  Rationale: packer's errors shall be managed beforehand by testing with 'packing-box test packer ...', its
            #             settings shall be fine-tuned BEFORE making datasets ; "unmanaged" errors should thus not occur
            if self._error:
                err = self._error.replace(str(exe) + ": ", "").replace(self.name + ": ", "").strip()
                self.logger.debug(f"not packed ({err})")
                return NOT_PACKED
            # if packed file's hash was not changed then change packer's state to "BAD" ; this will trigger a count at
            #  the dataset level and disable the packer if it triggers too many failures
            elif h == exe.hash:
                self.logger.debug("not packed (content not changed)")
                self._bad = True
                return NOT_PACKED
            # same if custom failure conditions are met ; packer's state is set to "BAD" and a counter is incremented
            #  for further disabling it if needed
            elif any(getattr(exe, a, None) == v for a, v in getattr(self, "failure", {}).items()):
                for a, v in self.failure.items():
                    if getattr(exe, a, None) == v:
                        self.logger.debug(f"not packed (failure condition met: {a}={v})")
                self._bad = True
                return NOT_LABELLED
            # it may occur that a packer modifies the format after packing, e.g. GZEXE on /usr/bin/fc-list (the new
            #  format becomes "POSIX shell script executable (binary data)'
            # this shall be simply discarded
            elif fmt != exe.format:
                self.logger.debug("packed but file type changed after packing")
                return label
            # if packing succeeded, we can return packer's label
            self.logger.debug("packed successfully")
            # if relevant, apply alterations based on defined Alterations
            alterations = getattr(self, "alterations", [])
            if len(alterations) > 0:
                from ..core.alterations import Alterations
                Alterations(exe, alterations)
                self.logger.debug("alterations applied:\n%s- " % "\n- ".join(alterations))
            return label
        
        def run(self, executable, **kwargs):
            """ Customizable method for shaping the command line to run the packer on an input executable. """
            # inspect steps and set custom parameters for non-standard packers
            p_pat = re.compile(re.sub(r"\)\?\}\}$", ")}}", PARAM_PATTERN.pattern))
            for step in getattr(self, "steps", [f"{self.name} {executable}"]):
                if "{{password}}" in step and 'password' not in self._params:
                    self._params['password'] = [random.randstr()]
                for name, value in p_pat.findall(step):
                    if name not in self._params:
                        self._params[name] = [" " + x if x.startswith("-") else _parse_parameter(x) \
                                              for x in value.split("|")]
            # then execute parent run() method taking the parameters into account
            return super(Packer, self).run(executable, **kwargs)
    # ensure it initializes only once (otherwise, this loops forever)
    if not __initialized:
        __initialized = True
        # dynamically makes Packer's registry of child classes from the default dictionary of packers
        #  (~/.packing-box/conf/packers.yml)
        Packer.source = None
    return Packer
Packer = lazy_object(__init)


# ------------------------------------------------ NON-STANDARD PACKERS ------------------------------------------------
def __init_ezuri():
    Packer = __init()

    class Ezuri(Packer):
        key = None
        iv  = None
        
        def run(self, executable, **kwargs):
            """ This packer prompts for parameters. """
            P = subprocess.PIPE
            p = subprocess.Popen(["ezuri"], stdout=P, stderr=P, stdin=P)
            executable = Executable(executable)
            self.logger.debug(f"inputs: src/dst={executable}, procname={executable.stem}")
            out, err = p.communicate((f"{executable}\n{executable}\n{executable.stem}\n"
                                      f"{'' if Ezuri.key is None else Ezuri.key}\n"
                                      f"{'' if Ezuri.iv is None else Ezuri.iv}\n").encode())
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
            return f"{self.name}[key:{Ezuri.key};iv:{Ezuri.iv}]"
    return Ezuri
Ezuri = lazy_object(__init_ezuri)

