# -*- coding: UTF-8 -*-
import json
import logging
import re
from argparse import ArgumentParser, RawTextHelpFormatter
from ast import literal_eval
from contextlib import suppress
from itertools import product
from os.path import abspath, exists, expanduser, isfile
from pprint import pformat
from shlex import split
from subprocess import call, Popen, PIPE
from sys import argv, exit
from time import perf_counter
from yaml import safe_load


__all__ = ["catch_exception", "json", "literal_eval", "pformat", "re", "run", "suppress", "version",
           "DETECTORS", "PACKERS"]


DETECTORS      = None
DETECTORS_FILE = "~/.packing-box/conf/detectors.yml"
PACKERS        = None
PACKERS_FILE   = "~/.packing-box/conf/packers.yml"

NOT_LABELLED, NOT_PACKED = "?-"

# source: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository
LICENSES = {
    'afl-3.0': "Academic Free License v3.0",
    'agpl-3.0': "GNU Affero General Public License v3.0",
    'apache-2.0': "Apache license 2.0",
    'artistic-2.0': "Artistic license 2.0",
    'bsd-2-clause': "BSD 2-clause \"Simplified\" license",
    'bsd-3-clause': "BSD 3-clause \"New\" or \"Revised\" license",
    'bsd-3-clause-clear': "BSD 3-clause Clear license",
    'bsl-1.0': "Boost Software License 1.0",
    'cc': "Creative Commons license family",
    'cc-by-4.0': "Creative Commons Attribution 4.0",
    'cc-by-sa-4.0': "Creative Commons Attribution Share Alike 4.0",
    'cc0-1.0': "Creative Commons Zero v1.0 Universal",
    'ecl-2.0': "Educational Community License v2.0",
    'epl-1.0': "Eclipse Public License 1.0",
    'eupl-1.1': "European Union Public License 1.1",
    'gpl': "GNU General Public License family",
    'gpl-2.0': "GNU General Public License v2.0",
    'gpl-3.0': "GNU General Public License v3.0",
    'isc': "ISC",
    'lgpl': "GNU Lesser General Public License family",
    'lgpl-2.1': "GNU Lesser General Public License v2.1",
    'lgpl-3.0': "GNU Lesser General Public License v3.0",
    'lppl-1.3c': "LaTeX Project Public License v1.3c",
    'mit': "MIT",
    'mpl-2.0': "Mozilla Public License 2.0",
    'ms-pl': "Microsoft Public License",
    'ncsa': "University of Illinois/NCSA Open Source License",
    'ofl-1.1': "SIL Open Font License 1.1",
    'osl-3.0': "Open Software License 3.0",
    'postgresql': "PostgreSQL License",
    'unlicense': "The Unlicense",
    'wtfpl': "Do What The F*ck You Want To Public License",
    'zlib': "zLib License",
}


def catch_exception(f):
    """ Decorator for returning function's result and an exception if relevant. """
    def _wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs), None
        except Exception as e:
            return None, str(e)
    return _wrapper


def execute(name, **kwargs):
    """ Run an OS command. """
    # from the specific arguments' parsed values and actions, reconstruct the options string
    spec, spec_val = "", kwargs.get('orig_args', {})
    for a in kwargs.get('_orig_args', []):
        n = a.dest
        v = spec_val[n]
        if a.__class__.__name__ == "_StoreTrueAction":
            if v is True:
                spec += " " + a.option_strings[0]
        elif a.__class__.__name__ == "_StoreFalseAction":
            if v is False:
                spec += " " + a.option_strings[0]
        elif a.__class__.__name__ == "_SubParsersAction":
            spec += " " + n
        elif isinstance(v, (list, tuple)):
            for x in v:
                spec += " " + a.option_strings[0] + " " + str(x)
        elif v is not None:
            spec += " " + a.option_strings[0] + " " + str(v)
    cmd = DETECTORS[name].get('command', "%s/%s {path}" % (expanduser("~/.local/bin"), name.lower()))
    exe, opt = cmd.replace("$OPT", expanduser("~/.opt/detectors")).replace("$BIN", expanduser("~/.opt/bin")) \
                  .split(" ", 1)
    cmd = (exe + "%s " + opt) % spec
    cmd = re.sub("'?\{path\}'?", "'{path}'", cmd)  # this allows to handle input path with whitespaces
    kwargs['logger'].debug("Command format: " + cmd)
    if kwargs.get('version', False):
        if kwargs.get('exit', True):
            call([cmd.split()[0], "--version"])
            exit(0)
        else:
            cmd = "%s --version" % cmd.split()[0]
    shell = ">" in cmd
    # prepare the command line and run the tool
    cmd = cmd.format(**kwargs)
    cmd = cmd if shell else split(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell)
    out, err = proc.communicate()
    return out.decode(errors="ignore"), err.decode(errors="ignore")


def normalize(*lines, **kwargs):
    """ Normalize the output from a list of values based on the PACKERS list according to a decision heuristic.
    
    :param lines: selection of detector's output lines
    """
    if len(lines) == 0 or lines in [(None, ), ("", )]:
        return NOT_PACKED
    # count results
    d, unknown = {}, 0
    for l in lines:
        b = False
        for packer, details in PACKERS.items():
            # compose the name-based pattern and append the list of aliases
            patterns = ["(?i)(?:[^a-z]|^)%s(?:[^a-z]|$)" % packer.replace("_", "(?:[ -_]|)")] + \
                       details.get('aliases', [])
            # use the line as is but also a sanitized version
            strings  = [l, re.sub(r"(\'s|\s|\(|\)|\(\-\)|\[\-\])", "", l)]
            for pattern, string in product(patterns, strings):
                if re.search(pattern, string):
                    p = packer.lower()
                    d.setdefault(p, 0)
                    d[p] += 1
                    b = True
                    break
        # collected strings that could not be matched by a packer name or related pattern are considered suspicious
        if not b:
            unknown += 1
    # if no detection, check for suspicions and return "unknown" if found, otherwise return NOT_PACKED
    if len(d) == 0:
        if unknown > 0:
            kwargs['logger'].debug("Suspicions:\n- %s" % "\n- ".join(strings))
            return "unknown"
        return NOT_PACKED
    vmax = max(d.values())
    m = [k for k, v in d.items() if v == vmax]
    kwargs['logger'].debug("Matches: %s\n" % d)
    # trivial when no candidate ; consider unknown count
    if len(m) == 0:
        if unknown > 0:
            return "unknown"
        return NOT_PACKED
    # trivial when there is only one maxima
    elif len(m) == 1:
        return m[0]
    # when multiple maxima, only decide if longest match AND shorter strings are include in the longest match ;
    #  otherwise, return NOT_LABELLED
    else:
        best = m[0]
        for s in m[1:]:
            if s in best or best in s:
                best = max([s, best], key=len)
            else:
                # undecided
                return NOT_LABELLED
        return best


def run(name, exec_func=execute, parse_func=lambda x, **kw: x, stderr_func=lambda x, **kw: x, parser_args=[],
        normalize_output=True, binary_only=False, parse_stderr=False, weak_assumptions=False, **kwargs):
    """ Run a tool and parse its output.
    
    It also allows to parse stderr and to normalize the output.
    
    :param name:             name of the tool
    :param exec_func:        function for executing the tool
    :param parse_func:       function for parsing the output of stdout
    :param stderr_func:      function for handling the output of stderr
    :param parser_args:      additional arguments for the parser ; format: [(args, kwargs), ...]
    :param normalize_output: normalize the final output based on a base of items
    :param binary_only:      specify that the tool only handles binary classes (i.e. no packer name)
    :param weak_assumptions: specify that the tool has options depending on weak assumptions (e.g. suspicions)
    
    The parse_func shall take the output of stdout and return either a parsed value or None (if no relevant result).
    The stderr_func shall take the output of stderr and return either a parsed error message or None (if no error).
    """
    global DETECTORS, DETECTORS_FILE, PACKERS, PACKERS_FILE
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, add_help=False)
    opt = parser.add_argument_group("decision arguments")
    exe_type = kwargs.pop('exe_type', "exe")
    parser.add_argument(exe_type, help=kwargs.pop('exe_help', "path to executable"))
    # handle binary-only tools
    if binary_only:
        normalize_output = False
        with suppress(ValueError):
            argv.remove("--binary")
    else:
        opt.add_argument("--binary", action="store_true", help="output yes/no instead of packer's name")
    # handle weak assumption, e.g. when a tool can output detections and suspicions ; when setting --weak, it will also
    #  consider suspicions for the final decision
    if weak_assumptions:
        opt.add_argument("--weak", action="store_true", help="use weak assumptions for processing")
    else:
        with suppress(ValueError):
            argv.remove("--weak")
    # handle other specific options
    spec, spec_opt = parser.add_argument_group("original arguments"), []
    for args, kw in parser_args:
        spec_opt.append(spec.add_argument(*args, **kw))
    # handle help options
    if "--version" in argv:
        argv[1:] = ["DOESNOTEXIST", "--version"]
    extra = parser.add_argument_group("extra arguments")
    extra.add_argument("-b", "--benchmark", action="store_true", help="enable benchmarking")
    extra.add_argument("-h", "--help", action="help", help="show this help message and exit")
    extra.add_argument("-v", "--verbose", action="store_true", help="display debug information")
    extra.add_argument("--version", action="store_true", help="show version and exit")
    extra.add_argument("--detectors-file", default=DETECTORS_FILE, help="path to detectors YAML")
    if normalize_output:  # the PACKERS list is only required when normalizing
        extra.add_argument("--packers-file", default=PACKERS_FILE, help="path to packers YAML")
    a = parser.parse_args()
    # put parsed values and specific arguments' actions in dedicated namespace variables
    a.orig_args, a._orig_args = {}, spec_opt
    for opt in spec_opt:
        n = opt.dest
        try:
            a.orig_args[n] = expanduser(getattr(a, n))
        except TypeError:
            a.orig_args[n] = getattr(a, n)
        delattr(a, n)
    if binary_only:
        a.binary = True
    logging.basicConfig(format="[%(levelname)s] %(message)s")
    a.logger = logging.getLogger(name.lower())
    a.logger.setLevel([logging.INFO, logging.DEBUG][a.verbose])
    p = a.path = abspath(getattr(a, exe_type))
    if getattr(a, exe_type) != "DOESNOTEXIST" and not exists(p):
        a.logger.error("file not found")
        exit(1)
    # load related dictionaries
    DETECTORS_FILE = a.detectors_file
    with open(expanduser(DETECTORS_FILE)) as f:
        DETECTORS = safe_load(f.read())
    if normalize_output:
        PACKERS_FILE = a.packers_file
        with open(expanduser(PACKERS_FILE)) as f:
            PACKERS = safe_load(f.read())
    # handle version display
    if a.version:
        v = DETECTORS[name].get('version')
        if v:
            # accepted formats:
            #  - module.submodules:variable ; e.g. peid.__info__:__version__
            p = v.split(":")
            if len(p) == 2:
                with suppress(Exception):
                    sp = p[0].split(".")
                    m, i = __import__(sp[0]), 1
                    while i < len(sp):
                        m = getattr(m, sp[i])
                        i += 1
                    v = getattr(m, p[1])
            else:
                #  - <output> => get the version from the output
                if v == "<output>":
                    out, err = execute(name, version=True, exit=False, logger=a.logger)
                    out += err
                #  - file path ; e.g. ~/.opt/detectors/detector_name/version.txt
                elif isfile(v):
                    with open(v) as f:
                        out = f.read().strip()
                v = re.search(r"\d{1,2}(?:\.\d+){1,3}", out).group()
            #  - if not the first or second format, consider a string
            v = "%s %s" % (name, v.lstrip("v"))
            with suppress(KeyError):
                v += " <%s>" % DETECTORS[name]['source']
            with suppress(KeyError):
                v += " by " + DETECTORS[name]['author']
            with suppress(KeyError):
                l = DETECTORS[name]['license']
                v += "\nLicensed under %s (https://choosealicense.com/licenses/%s)" % (LICENSES[l], l)
            print(v)
            exit(0)
        exec_func(name, version=True, data=DETECTORS[name], logger=a.logger)
    # execute the tool
    t1 = perf_counter()
    out, err = exec_func(name, **vars(a))
    dt = perf_counter() - t1
    # now handle the result if no error
    err = stderr_func(err, **vars(a))
    msg = DETECTORS[name].get('silent')
    if msg is not None:
        err = "\n".join([line for line in (err or "").splitlines() if all(re.search(m, line) is None for m in msg)])
    if parse_stderr:
        out += "\n" + err
        err = ""
    if err:
        a.logger.error(err)
        exit(1)
    else:
        p = parse_func(out, **vars(a))
        if a.verbose and len(out) > 0:
            a.logger.debug("Output:\n" + ("\n".join(out) if isinstance(out, (list, tuple)) else \
                                          json.dumps(out, indent=4) if isinstance(out, dict) else str(out)) + "\n")
        if normalize_output:
            if not isinstance(p, list):
                p = [p]
            p = normalize(*p, **vars(a))
            if a.binary:
                # when NOT_LABELLED, this means that there was as many traces for multiple packers, hence it could not
                #  be decided but yet, it is packed
                p = str({NOT_PACKED: NOT_PACKED, None: None}.get(p, True))
        else:
            p = " ".join(p) if isinstance(p, (list, tuple)) else str(p)
        if p is not None:
            if a.benchmark:
                p += " " + str(dt)
            print(p)


def version(string):
    """ Decorator for handling a version string. """
    def _wrapper(f):
        def _subwrapper(*args, **kwargs):
            if kwargs.get('version', False):
                print(string)
                exit(0)
            return f(*args, **kwargs)
        return _subwrapper
    return _wrapper

