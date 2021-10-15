# -*- coding: UTF-8 -*-
import json
import logging
import re
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from ast import literal_eval
from os.path import abspath, exists
from pprint import pformat
from shlex import split
from subprocess import Popen, PIPE
from time import perf_counter
from yaml import safe_load


__all__ = ["json", "literal_eval", "pformat", "re", "run", "PACKERS", "PACKERS_FILE"]


DETECTORS      = None
DETECTORS_FILE = "/opt/detectors.yml"
PACKERS        = None
PACKERS_FILE   = "/opt/packers.yml"


def execute(name, **kwargs):
    """ Run an OS command. """
    cmd = DETECTORS[name].get('command', "/usr/bin/%s {path}" % name.lower())
    shell = ">" in cmd
    # prepare the command line and run the tool
    cmd = cmd.format(**kwargs)
    cmd = cmd if shell else split(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=shell)
    out, err = proc.communicate()
    return out.decode(), err.decode()


def normalize(*packers):
    """ Normalize the output from a list of values based on the PACKERS list.
    
    :param packers: list of packer-related strings
    """
    if len(packers) == 0 or packers in [(None, ), ("", )]:
        return
    d = {'unknown': -1}
    for s in packers:
        for p in map(lambda x: x.lower(), PACKERS.keys()):
            if re.search(p, s.lower()):
                d.setdefault(p, 0)
                d[p] += 1
    return max(d, key=d.get)


def run(name, exec_func=execute, parse_func=lambda x: x, stderr_func=lambda x: x, parser_args=[], db=False,
        normalize_output=True, binary_only=False, **kwargs):
    """ Run a tool and parse its output.
    
    It also allows to parse stderr and to normalize the output.
    
    :param name:             name of the tool
    :param exec_func:        function for executing the tool
    :param parse_func:       function for parsing the output of stdout
    :param stderr_func:      function for handling the output of stderr
    :param parser_args:      additional arguments for the parser ; format: [(args, kwargs), ...]
    :param db:               add a --db argument with the given value as default (do not add it if db=False)
    :param normalize_output: normalize the final output based on a base of items
    
    The parse_func shall take the output of stdout and return either a parsed value or None (if no relevant result).
    The stderr_func shall take the output of stderr and return either a parsed error message or None (if no error).
    """
    global DETECTORS, DETECTORS_FILE, PACKERS, PACKERS_FILE
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, add_help=False)
    opt, extra = parser.add_argument_group("optional arguments"), parser.add_argument_group("extra arguments")
    exe_type = kwargs.pop('exe_type', "exe")
    parser.add_argument(exe_type, help=kwargs.pop('exe_help', "path to executable"))
    if db is not False:
         opt.add_argument("-d", "--db", default=db, help="signatures database")
    if binary_only:
        normalize_output = False
        try:
            sys.argv.remove("--binary")
        except ValueError:
            pass
    else:
        opt.add_argument("--binary", action="store_true", help="output yes/no instead of packer's name")
    for args, kw in parser_args:
        opt.add_argument(*args, **kw)
    extra.add_argument("-b", "--benchmark", action="store_true", help="enable benchmarking")
    extra.add_argument("-h", "--help", action="help", help="show this help message and exit")
    extra.add_argument("-v", "--verbose", action="store_true", help="display debug information")
    extra.add_argument("--detectors-file", default=DETECTORS_FILE, help="path to detectors YAML")
    if normalize_output:  # the PACKERS list is only required when normalizing
        extra.add_argument("--packers-file", default=PACKERS_FILE, help="path to packers YAML")
    a = parser.parse_args()
    if binary_only:
        a.binary = True
    logging.basicConfig(format="[%(levelname)s] %(message)s")
    a.logger = logging.getLogger(name.lower())
    a.logger.setLevel([logging.INFO, logging.DEBUG][a.verbose])
    p = a.path = abspath(getattr(a, exe_type))
    if not exists(p):
        print("[ERROR] file not found")
        return
    # load related dictionaries
    DETECTORS_FILE = a.detectors_file
    with open(DETECTORS_FILE) as f:
        DETECTORS = safe_load(f.read())
    if normalize_output:
        PACKERS_FILE = a.packers_file
        with open(PACKERS_FILE) as f:
            PACKERS = safe_load(f.read())
    # execute the tool
    t1 = perf_counter()
    out, err = exec_func(name, **vars(a))
    dt = perf_counter() - t1
    # now handle the result if no error
    err = stderr_func(err)
    if err:
        a.logger.error(err)
    else:
        p = parse_func(out)
        if a.verbose and len(out) > 0 and p != out:
            a.logger.debug("\n" + out + "\n")
        if normalize_output:
            if not isinstance(p, list):
                p = [p]
            p = normalize(*p)
            if a.binary:
                p = str(p is not None)
        if p is not None:
            if a.benchmark:
                p += " " + str(dt)
            print(p)

