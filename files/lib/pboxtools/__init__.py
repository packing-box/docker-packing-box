# -*- coding: UTF-8 -*-
import json
import pefile
import peutils
import re
from argparse import ArgumentParser, RawTextHelpFormatter
from os.path import abspath, exists
from shlex import split
from subprocess import Popen, PIPE
from sys import stderr
from time import perf_counter
from yaml import safe_load


__all__ = ["json", "pefile", "peutils", "re", "run", "PACKERS", "PACKERS_FILE"]


DETECTORS      = None
DETECTORS_FILE = "/opt/detectors.yml"
PACKERS        = None
PACKERS_FILE   = "/opt/packers.yml"


def execute(name, **kwargs):
    """ Run an OS command. """
    cmd = DETECTORS[name]['command']
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
    if packers == (None, ):
        return
    d = {'unknown': -1}
    for s in packers:
        for p in map(lambda x: x.lower(), PACKERS.keys()):
            if re.search(p, s):
                d.setdefault(p, 0)
                d[p] += 1
    return max(d, key=d.get)


def run(name, exec_func=execute, parse_func=lambda x: x, stderr_func=lambda x: None, normalize_output=True, **kwargs):
    """ Run a tool and parse its output.
    
    It also allows to parse stderr and to normalize the output.
    
    :param name:             name of the tool
    :param exec_func:        function for executing the tool
    :param parse_func:       function for parsing the output of stdout
    :param stderr_func:      function for handling the output of stderr
    :param shell:            shell option for Popen
    :param normalize_output: normalize the final output based on a base of items
    
    The parse_func shall take the output of stdout and return either a parsed value or None (if no relevant result).
    The stderr_func shall take the output of stderr and return either a parsed error message or None (if no error).
    """
    global DETECTORS, DETECTORS_FILE, PACKERS, PACKERS_FILE
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    exe_type = kwargs.pop('exe_type', "exe")
    parser.add_argument(exe_type, help=kwargs.pop('exe_help', "path to executable"))
    parser.add_argument("-b", "--benchmark", action="store_true", help="enable benchmarking")
    if "db" in kwargs.keys():
        parser.add_argument("-d", dest="db", default=kwargs['db'], help=kwargs.get('db_help', "signatures database"))
    parser.add_argument("-v", "--verbose", action="store_true", help="display debug information")
    parser.add_argument("--detectors-file", default=DETECTORS_FILE, help="path to detectors YAML")
    parser.add_argument("--packers-file", default=PACKERS_FILE, help="path to packers YAML")
    a = parser.parse_args()
    p = kwargs['path'] = abspath(getattr(a, exe_type))
    if not exists(p):
        print("[ERROR] file not found")
        return
    # load related dictionaries
    DETECTORS_FILE = a.detectors_file
    with open(DETECTORS_FILE) as f:
        DETECTORS = safe_load(f.read())
    PACKERS_FILE = a.packers_file
    with open(PACKERS_FILE) as f:
        PACKERS = safe_load(f.read())
    # execute the tool
    t1 = perf_counter()
    out, err = exec_func(name, **kwargs)
    dt = perf_counter() - t1
    # now handle the result
    if a.verbose:
        stderr.write("[DEBUG]\n" + out)
    err = stderr_func(err)
    if err is not None:
        stderr.write("[ERROR]\n" + err)
    else:
        p = parse_func(out)
        if not isinstance(p, list):
            p = [p]
        if normalize_output:
            p = normalize(*p)
        if p is not None:
            if a.benchmark:
                p += " " + str(dt)
            print(p)

