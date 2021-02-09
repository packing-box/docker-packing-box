#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import logging
import shutil
from tinyscript.helpers import yaml_config, Path


__all__ = []  # this list is filled in at the end of this module


CATEGORIES = {
    'All':    ["ELF", "Mach-O", "MSDOS", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     ["PE32", "PE64"],
}
SIGNATURES = {
    '^Mach-O 32-bit ':           "Mach-O32",
    '^Mach-O 64-bit ':           "Mach-O64",
    '^Mach-O universal binary ': "Mach-Ou",
    '^MS-DOS executable ':       "MSDOS",
    '^PE32 executable ':         "PE32",
    '^PE32\+ executable ':       "PE64",
    '^(set[gu]id )?ELF 32-bit ': "ELF32",
    '^(set[gu]id )?ELF 64-bit ': "ELF64",
}


def exe_format(executable):
    for ftype, fmt in SIGNATURES.items():
        if ts.is_filetype(str(executable), ftype):
            return fmt


class Packer:
    """ Packer abstraction, suitable for subclassing and redefining its .cmd() method for shaping its command line. """
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.logger = logging.getLogger(self.name)
        logging.setLoggers(self.name)
        self.enabled = shutil.which(self.name) is not None
        self.categories = Packer.expand(*self.categories)
    
    def cmd(self, executable, **kwargs):
        """ Customizable method for shaping the command line to run the packer on an input executable. """
        return [self.name, executable], self.name
    
    def pack(self, executable, **kwargs):
        """ Runs the packer according to its command line format and checks if the executable has been changed by this
             execution. """
        # check 1: does packer's binary exist ?
        if not self.enabled:
            self.logger.warning("%s disabled" % self.name)
            return
        # check 2: is packer selected while using the pack() method ?
        if not any(c in self.categories for c in Packer.expand(*kwargs.get('categories', ["all"]))):
            self.logger.debug("%s not selected" % self.name)
            return
        # check 3: is input executable in an applicable format ?
        fmt = exe_format(executable)
        if fmt is None:
            return
        if fmt not in self.categories:
            self.logger.debug("%s cannot be packed by %s" % (executable, self.name))
            return
        # now pack the input executable, taking its SHA256 in order to check for changes
        s256 = hashlib.sha256_file(executable)
        cmd, label = self.cmd(executable, **kwargs)
        out, err, retc = ts.execute(cmd, returncode=True)
        if retc:
            self.logger.error(ensure_str(err).strip())
            return
        if s256 == hashlib.sha256_file(executable):
            self.logger.warning("%s's content was not changed by %s" % (executable, self.name))
            return
        # if packing succeeded, we can return packer's label
        self.logger.debug("%s packed with %s" % (executable, self.name))
        return label
    
    @staticmethod
    def expand(*categories):
        """ 2-depth dictionary-based expansion function for resolving a list of executable categories. """
        selected = []
        for c in categories:                    # depth 1: e.g. All => ELF,PE OR ELF => ELF32,ELF64
            for sc in CATEGORIES.get(c, [c]):   # depth 2: e.g. ELF => ELF32,ELF64
                for ssc in CATEGORIES.get(sc, [sc]):
                    if ssc not in selected:
                        selected.append(ssc)
        return selected


# THIS PACKER DOES NOT SEEM TO WORK
#class M0dern_P4cker(Packer):
#    categories = ["elf"]
#    source = "https://github.com/n4sm/m0dern_p4cker"
#    
#    def cmd(self, executable, **kwargs):
#        """ This packer allows to define 3 different stubs: XOR, NOT, XORP (see documentation). """
#        stub = random.choice(["xor", "not", "xorp"])
#        return [self.name, executable, stub], "%s[%s]" % (self.name, stub)


# dynamically makes Packer child classes from the PACKERS dictionary
for packer, data in yaml_config(str(Path("packers.yml"))).items():
    if packer not in globals():
        p = globals()[packer] = type(packer, (Packer, ), dict(Packer.__dict__))
    else:
        p = globals()[packer]
    for k, v in data.items():
        setattr(p, k, v)
    __all__.append(packer)

