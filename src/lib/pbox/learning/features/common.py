# -*- coding: UTF-8 -*-
import ast
import lief
import os
from bintropy import entropy
from pygments.lexer import RegexLexer, bygroups, using
from pygments.token import Error, Keyword, Name, Number, Operator, String, Whitespace


__all__ = ["block_entropy", "block_entropy_per_section", "entropy", "section_characteristics",
           "_FEATURE_OPERATIONS", "_FEATURE_TRANSFORMERS"]


_FEATURE_OPERATIONS = {
    # single input operations
    'is_0':       lambda x: x == 0,
    'is_eq':      lambda x, v: x == v,
    'is_in':      lambda x, *v: x in v,
    'is_gt':      lambda x, f: x > f,
    'is_gte':     lambda x, f: x >= f,
    'is_lt':      lambda x, f: x < f,
    'is_lte':     lambda x, f: x <= f,
    'is_mult':    lambda x, i: x % i == 0,
    'is_not_in':  lambda x, *v: x not in v,
    'is_not_0':   lambda x: x != 0,
    'is_within':  lambda x, v1, v2: v1 <= x <= v2,
    'multiplier': lambda x, m: x / m if x % m == 0 else -1,
    'threshold':  lambda x, t, incl=True: x >= t if incl else x > t,
    # list-applicable operations
    'mean':       lambda l: mean(l),
}
_FEATURE_OPERATIONS['is_equal'] = _FEATURE_OPERATIONS['is_eq']
with open("/opt/features.yml") as f:
    _FEATURE_TRANSFORMERS = yaml.load(f, Loader=yaml.Loader)


# selection of in-scope characteristics of Section objects for an executable parsed by LIEF
CHARACTERISTICS = ["characteristics", "entropy", "numberof_line_numbers", "numberof_relocations", "offset", "size",
                   "sizeof_raw_data", "virtual_size"]


def _parse_binary(f):
    def _wrapper(executable, *args, **kwargs):
        # try to parse the binary first ; capture the stderr messages from LIEF
        tmp_fd, null_fd = os.dup(2), os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 2)
        binary = lief.parse(str(executable))
        os.dup2(tmp_fd, 2)  # restore stderr
        os.close(null_fd)
        if binary is None:
            raise OSError("Unknown format")
        return f(binary, *args, **kwargs)
    return _wrapper


# compute (name, data) pair for a Section object
_chars = lambda s: (s.name, {k: getattr(s, k) for k in CHARACTERISTICS})
# compute (name, entropy) pair for a Section object
_entr = lambda s, bs=0, z=False: (s.name, entropy(s.content, bs, z))

block_entropy = lambda bsize: lambda exe: str(entropy(exe.read_bytes(), bsize, True))
block_entropy_per_section = lambda bsize: _parse_binary(lambda exe: str([_entr(s, bsize, True) for s in exe.sections]))
section_characteristics   = _parse_binary(lambda exe: str([_chars(s) for s in exe.sections]))

