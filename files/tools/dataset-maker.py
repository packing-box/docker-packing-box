#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import json
from tinyscript import *


__author__    = "Alexandre D'Hondt"
__email__     = "alexandre.dhondt@gmail.com"
__version__   = "1.0.0"
__copyright__ = ("A. D'Hondt", 2021)
__license__   = "gpl-3.0"
__doc__       = """
This tool aims to make a dataset from executables contained in the packing-box Docker image or from a user-defined
 source of executable files, packed or not with the selected packers installed in the image.
"""
__examples__  = [
    "-c dotnet -n 1000",
    "-c pe,dotnet -n 2000",
]


CATEGORIES = {
    'all':   ["elf", "macho", "msdos", "pe"],
    'elf':   ["elf32", "elf64"],
    'macho': ["macho32", "macho64", "macho-u"],
    'pe':    ["pe32", "pe64"],
}
PACKERS = {
    'Amber':         ["pe"],
    'APack':         ["pe"],
    'Ezuri':         ["elf"],
    'Kkrunchy':      ["pe"],
    'PEtite':        ["pe"],
    'UPX':           ["elf", "mac", "pe"],
}
PACKING_BOX_SOURCES = ["/bin", "/usr/bin", "~/.wine/drive_c", "~/.wine32/drive_c"]
SIGNATURES = {
    '^Mach-O 32-bit ':           "macho32",
    '^Mach-O 64-bit ':           "macho64",
    '^Mach-O universal binary ': "macho-u",
    '^MS-DOS executable ':       "msdos",
    '^PE32 executable ':         "pe32",
    '^PE32\+ executable ':       "pe64",
    '^(set[gu]id )?ELF 32-bit ': "elf32",
    '^(set[gu]id )?ELF 64-bit ': "elf64",
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
        for c in categories:                    # depth 1: e.g. all => elf,pe OR elf => elf32,elf64
            for sc in CATEGORIES.get(c, [c]):   # depth 2: e.g. elf => elf32,elf64
                for ssc in CATEGORIES.get(sc, [sc]):
                    if ssc not in selected:
                        selected.append(ssc)
        return selected


# THIS PACKER DOES NOT SEEM TO WORK
#class M0dern_P4cker(Packer):
#    categories = ["elf"]
#    
#    def cmd(self, executable, **kwargs):
#        """ This packer allows to define 3 different stubs: XOR, NOT, XORP (see documentation). """
#        stub = random.choice(["xor", "not", "xorp"])
#        return [self.name, executable, stub], "%s[%s]" % (self.name, stub)


# dynamically makes Packer child classes from the PACKERS dictionary
for packer, categories in PACKERS.items():
    globals()[packer] = type(packer, Packer.__bases__, dict(Packer.__dict__))
    globals()[packer].categories = categories


if __name__ == '__main__':
    parser.add_argument("-c", "--categories", type=ts.values_list, default="all",
                        help="list of categories to be considered")
    parser.add_argument("-s", "--destination-dir", default="dataset", type=ts.folder_does_not_exist,
                        help="executables destination directory for the dataset")
    parser.add_argument("-n", "--number-executables", dest="n", type=ts.pos_int, default=100,
                        help="number of executables for the output dataset")
    parser.add_argument("-s", "--source-dir", default=PACKING_BOX_SOURCES, nargs="*",
                        type=lambda p: ts.Path(p, create=True),
                        help="executables source directory to be included")
    initialize()
    args.categories = Packer.expand(*categories)
    packers = []
    for k, i in list(globals().items()):
        if k in PACKERS.keys() or ts.is_class(i) and i.__base__ is Packer:
            packer = i()
            if packer.enabled:
                packers.append(packer)
    if len(packers) == 0:
        logger.critical("No packer found")
    else:
        logger.info("Packers: %s" % ", ".join(sorted([p.name for p in packers])))
        candidates = []
        for src in args.source_dir:
            candidates.extend(ts.Path(src, expand=True).listdir())
        executables = {}
        random.shuffle(candidates)
        dst = ts.Path(args.destination_dir, create=True)
        for f in candidates:
            if exe_format(f) is None or f.filename in executables.keys():
                continue
            if len(executables) >= args.n:
                break
            df = dst.joinpath(f.filename)
            shutil.copy(str(f), str(df))
            df.chmod(0o777)
            label = None
            if random.randint(0, 1):
                label = random.choice(packers).pack(str(df), categories=args.categories)
                if label is None:
                    continue
            executables[f.filename] = label
        labels = dst.joinpath("labels.json")
        labels.touch()
        with labels.open('w') as f:
            json.dump(executables, f, indent=2)

