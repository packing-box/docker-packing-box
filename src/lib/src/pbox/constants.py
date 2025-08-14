# -*- coding: UTF-8 -*-
import builtins as bi
from subprocess import check_output
from tinyscript import colored, re, sys
from tinyscript.helpers import slugify, Path


bi.colored = colored
# basic framework constants
def __init__cpu_count():
    import multiprocessing as mp
    return mp.cpu_count()
bi.CPU_COUNT = lazy_load_object("CPU_COUNT", __init__cpu_count)
bi.EXPORT_FORMATS = {'md': "markdown", 'tex': "latex", 'txt': "string", 'xlsx': "excel"}
for f in ['csv', 'html', 'json', 'pickle', 'xml', 'yml']:
    bi.EXPORT_FORMATS[f] = f
bi.LOG_FORMATS = ["%(asctime)s [%(levelname)s] %(message)s", "%(asctime)s [%(levelname)-8s] %(name)-18s %(message)s"]
bi.PACKING_BOX_SOURCES = {
    'ELF': ["/usr/bin", "/usr/sbin"],
    'PE':  ["~/.wine32/drive_c/windows", "~/.wine64/drive_c/windows"],
}
bi.PBOX_HOME = Path("~/.packing-box", create=True, expand=True)
bi.RENAME_FUNCTIONS = {
    'as-is':   lambda p: p,
    'slugify': slugify,
}
bi.SPECIAL_INPUTS = ["ALL"]  # first element is for 'all', next ones are for eventual future use


# detection
bi.THRESHOLDS = {
    'absolute-majority': lambda l: round(l / 2. + .5),
}


# automation
# for a screenshot: "xwd -display $DISPLAY -root -silent | convert xwd:- png:screenshot.png"
bi.GUI_SCRIPT = """#!/bin/bash
source ~/.bash_xvfb
{{preamble}}
SRC="$1"
NAME="$(basename "$1" | sed 's/\(.*\)\..*/\1/')"
DST="$HOME/.wine%(arch)s/drive_c/users/user/Temp/${1##*/}"
FILE="c:\\\\users\\\\user\\\\Temp\\\\${1##*/}"
cp -f "$SRC" "$DST"
WINEPREFIX=\"$HOME/.wine%(arch)s\" WINEARCH=win%(arch)s wine "$EXE" &
sleep .5
{{actions}}
ps -eaf | grep -v grep | grep -E -e "/bin/bash.+bin/$NAME" -e ".+/$NAME\.exe\$" \
                                 -e 'bin/wineserver$' -e 'winedbg --auto' \
                                 -e 'windows\\system32\\services.exe$' \
                                 -e 'windows\\system32\\conhost.exe --unix' \
                                 -e 'windows\\system32\\explorer.exe /desktop$' \
        | awk {'print $2'} | xargs kill -9
sleep .1
mv -f "$DST" "$SRC"{{postamble}}
"""
bi.OS_COMMANDS = lazy_object(lambda: check_output("compgen -c", shell=True, executable="/bin/bash").splitlines())
bi.ERR_PATTERN = lazy_object(lambda: re.compile(r"^\x07?\s*(?:\-\s*)?(?:\[(?:ERR(?:OR)?|\!)\]|ERR(?:OR)?\:)\s*"))
bi.PARAM_PATTERN = lazy_object(lambda: re.compile(r"{{([^\{\}]*?)(?:\[([^\{\[\]\}]*?)\])?}}"))
bi.STATUS_DISABLED = ["broken", "commercial", "info", "useless"]
bi.STATUS_ENABLED = lazy_object(lambda: [s for s in STATUS.keys() if s not in STATUS_DISABLED + ["not installed"]])


# terminal
bi.ACRONYMS = ["cfg"]
bi.COLORMAP = {
    'red':        (255, 0,   0),
    'lightCoral': (240, 128, 128),
    'purple':     (128, 0,   128),
    'peru':       (205, 133, 63),
    'salmon':     (250, 128, 114),
    'rosyBrown':  (188, 143, 143),
    'sandyBrown': (244, 164, 96),
    'sienna':     (160, 82,  45),
    'plum':       (221, 160, 221),
    'pink':       (255, 192, 203),
    'tan':        (210, 180, 140),
    'tomato':     (255, 99,  71),
    'violet':     (238, 130, 238),
    'magenta':    (255, 0,   255),
    'fireBrick':  (178, 34,  34),
    'indigo':     (75,  0,   130),
}
bi.DEFAULT_BACKEND = "rich"

def __init(*args):
    def _wrapper():
        if isinstance(args[0], dict):
            return {k: colored(*v) if isinstance(v, tuple) else v for k, v in args[0].items()}
        return colored(*args)
    return _wrapper

bi.NOK      = lazy_object(__init("☒", "red"))
bi.NOK_GREY = lazy_object(__init("☒", "grey50"))
bi.OK       = lazy_object(__init("☑", "green"))
bi.OK_GREY  = lazy_object(__init("☑", "grey50"))
bi.STATUS   = lazy_object(__init({
                            'broken':        ("☒", "magenta"),
                            'commercial':    "  💰",
                            'gui':           ("🗗", "cyan"),
                            'info':          ("ⓘ", "grey"),
                            'installed':     ("☑", "orange"),
                            'not installed': ("☒", "red"),
                            'ok':            ("☑", "green"),
                            'test':          ("☐", "grey"),
                            'todo':          ("☐", "grey"),
                            'useless':       ("ⓘ", "grey"),
                        }))


# binary parsing
bi.DEFAULT_SECTION_SLOTS = ["name", "size", "offset", "content", "virtual_address"]
bi.DEFAULT_SEGMENT_SLOTS = ["content", "flags", "virtual_address", "virtual_size"]
bi.RECURSION_LIMIT = sys.getrecursionlimit()


# executable
bi.ANGR_ENGINES = ["default", "pcode", "vex"]
bi.CFG_ALGORITHMS = ["emulated", "fast"]
bi.DATA_EXTENSIONS = [".json", ".txt"]
bi.EXE_METADATA = ["realpath", "format", "signature", "size", "ctime", "mtime"]
bi.FORMATS = {
    'All':    ["ELF", "Mach-O", "PE"],
    'ELF':    ["ELF32", "ELF64"],
    'Mach-O': ["Mach-O32", "Mach-O64", "Mach-Ou"],
    'PE':     [".NET", "MSDOS", "PE32", "PE64"],
}
bi.IMPORT_SUFFIXES = {
    'default': (),
    'PE': ('', 'A', 'W', 'Ex', 'ExA', 'ExW'),
}
bi.SIGNATURES = {
    '^Mach-O 32-bit ':                         "Mach-O32",
    '^Mach-O 64-bit ':                         "Mach-O64",
    '^Mach-O universal binary ':               "Mach-Ou",
    '^MS-DOS executable\s*':                   "MSDOS",
    '^PE32\+? executable (.+?)\.Net assembly': ".NET",
    '^PE32 executable ':                       "PE32",
    '^PE32\+ executable ':                     "PE64",
    '^(set[gu]id )?ELF 32-bit ':               "ELF32",
    '^(set[gu]id )?ELF 64-bit ':               "ELF64",
}
bi.TEST_FILES = {
    'ELF32': [
        "/usr/bin/perl",
        "/usr/lib/wine/wine",
        "/usr/lib/wine/wineserver32",
        "/usr/libx32/crti.o",
        "/usr/libx32/libpcprofile.so",
    ],
    'ELF64': [
        "/bin/cat",
        "/bin/ls",
        "/bin/mandb",
        "/usr/lib/openssh/ssh-keysign",
        "/usr/lib/git-core/git",
        "/usr/lib/x86_64-linux-gnu/crti.o",
        "/usr/lib/x86_64-linux-gnu/libpcprofile.so",
        "/usr/lib/ld-linux.so.2",
    ],
    'MSDOS': [
        "~/.wine32/drive_c/windows/rundll.exe",
        "~/.wine32/drive_c/windows/system32/gdi.exe",
        "~/.wine32/drive_c/windows/system32/user.exe",
        "~/.wine32/drive_c/windows/system32/mouse.drv",
        "~/.wine32/drive_c/windows/system32/winaspi.dll",
    ],
    'PE32': [
        "~/.wine32/drive_c/windows/winhlp32.exe",
        "~/.wine32/drive_c/windows/system32/plugplay.exe",
        "~/.wine32/drive_c/windows/system32/winemine.exe",
        "~/.wine32/drive_c/windows/twain_32.dll",
        "~/.wine32/drive_c/windows/twain_32/sane.ds",
        "~/.wine32/drive_c/windows/system32/msscript.ocx",
        "~/.wine32/drive_c/windows/system32/msgsm32.acm",
    ],
    'PE64': [
        "~/.wine64/drive_c/windows/hh.exe",
        "~/.wine64/drive_c/windows/system32/spoolsv.exe",
        "~/.wine64/drive_c/windows/system32/dmscript.dll",
        "~/.wine64/drive_c/windows/twain_64/gphoto2.ds",
        "~/.wine64/drive_c/windows/system32/msscript.ocx",
        "~/.wine64/drive_c/windows/system32/msadp32.acm",
    ],
}
bi.X86_64_JUMP_MNEMONICS = {"call", "jmp", "bnd jmp", "je", "jne", "jz", "jnz", "ja", "jae", "jb", "jbe", "jl", "jle",
                           "jg", "jge", "jo", "jno", "js", "jns", "jp", "jnp", "jecxz", "jrcxz", "jmpf", "jmpq", "jmpw"}
bi.X86_64_REGISTERS = {
    "Return value":              {"rax", "eax", "ax", "ah", "al"},
    "General-Purpose Registers": {"rbx", "rcx", "rdx", "ebx", "ecx", "edx", "bx", "bh", "bl", "cx", "ch", "cl", "dx",
                                  "dh", "dl"},
    "Segment Registers":         {"cs", "ds", "es", "fs", "gs", "ss"},
    "Function arguments":        {"rsi", "rdi", "esi", "edi"},
    "Stack Registers":           {"rbp", "rsp", "ebp", "esp"},
    "Instruction Pointer":       {"rip", "eip"},
    "Flags Register":            {"rflags", "eflags"},
    "Floating-Point Registers":  set(f"xmm{i}" for i in range(16))
}


# experiments
bi.COMMIT_VALID_COMMANDS = [
    # OS commands
    "cd", "cp", "mkdir", "mv",
    # packing-box commands
    "alteration", "analyzer", "dataset", "detector", "executable", "feature", "model", "packer", "unpacker",
    "visualizer",
]


# machine learning & visualization
bi.FEATURE_CONSTANTS = ["IMPORT_SUFFIXES", "X86_64_JUMP_MNEMONICS"]
bi.FEATURE_PTIME = ["low", "medium", "high", "extreme"]
bi.IMG_FORMATS = ("jpg", "png", "tif", "svg")
bi.LABELS = {
    'not-packed':         "Original",
    'Notpacked':          "Original",
    'BeRoEXEPacker':      "BeRo",
    'Enigma Virtual Box': "Enigma VBox",
    'Eronana Packer':     "Eronana",
}
bi.MARKERS = "v^<>.+x*ospxhd123"
# metric format function: p=precision, m=multiplier, s=symbol
_mformat = lambda p=3, m=1, s=None: lambda x: "-" if x == "-" else ("{:.%df}{}" % p).format(m * x, s or "")
bi.METRIC_DISPLAY = {
    # helpers
    '%':   _mformat(2, 100, "%"),
    'ms':  _mformat(3, 1000, "ms"), # used for 'Processing Time' in metric_headers(...)
    'nbr': _mformat(),
    'classification': {
        'Accuracy':  "%",
        'Precision': "%",
        'Recall':    "%",
        'F-Measure': "%",
        'MCC':       "%",
        'AUC':       "%",
    },
    'clustering': {
        # labels known
        'Rand\nScore': "nbr",
        'Adjusted\nMutual\nInformation': "nbr",
        'Homogeneity': "nbr",
        'Completeness': "nbr",
        'V-Measure': "nbr",
        # labels not known
        'Silhouette\nScore': "nbr",
        'Calinski\nHarabasz\nScore': "nbr",
        'Davies\nBouldin\nScore': "nbr",
    },
    'regression': {
        'MSE': "nbr",
        'MAE': "nbr",
    },
}
bi.NO_METRIC_VALUE = "-"
bi.UNDEF_RESULT = "undefined"
# label markers and conversion for Scikit-Learn and Weka
bi.NOT_LABELLED, bi.NOT_PACKED = "?-"  # impose markers for distinguishing between unlabelled and not-packed data
bi.LABELS_BACK_CONV = {NOT_LABELLED: -1, NOT_PACKED: 0}  # values used with sklearn for unlabelled and null class
bi.READABLE_LABELS = lambda l, binary=False: {(LABELS_BACK_CONV[NOT_PACKED] if binary else NOT_PACKED): 'not packed', \
                                        (LABELS_BACK_CONV[NOT_LABELLED] if binary else NOT_LABELLED): 'not labelled'} \
                                        .get(l, 'packed' if binary else l)

