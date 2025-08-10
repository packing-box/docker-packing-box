# -*- coding: UTF-8 -*-
from cffi import FFI
from tinyscript import functools, re, string, subprocess
from tinyscript.helpers import execute, TempPath


__all__ = ["compare_files", "compare_fuzzy_hashes", "mrsh_v2", "sdhash", "ssdeep", "tlsh"]


_FFI = FFI()
_FFI.cdef(
    """
    int fuzzy_hash_buf(const unsigned char *buf, unsigned int buf_len, char *result);
    int fuzzy_hash_filename(const char *filename, char *result);
    int fuzzy_compare(const char *s1, const char *s2);
    """
)
_FUZZY_LIB = _FFI.dlopen("libfuzzy.so")
_FUZZY_MAX_RESULT = 2048
_HASH_FORMATS_REGEX = {
    'tlsh':    re.compile(r"^[0-9A-F]{70}$"),
    'ssdeep':  re.compile(r"^\d+(?:\:[^\:]+){2}$"),
    'sdhash':  re.compile(r"^sdbf(?:\-[a-z]+)?(?:\:[^\:]+){12,}$"),
    'mrsh-v2': re.compile(r"^[^:]+(?:\:[^\:]+){4}$"),
}


def _hash_format(h):
    for a, r in _HASH_FORMATS_REGEX.items():
        if r.match(h):
            return a


def _new_cstr(py_bytes: bytes):
    return _FFI.new("char[]", py_bytes)


def _tool_wrapper(f):
    @functools.wraps(f)
    def __wrapper(path, **parameters):
        p = [[f"{['--', '-'][len(k) == 1]}{k.replace('_', '-')}", f"{v}"] for k, v in parameters.items()]
        out, err = execute([f.__name__.replace("_", "-")] + [x for l in p for x in l] + [str(path)])
        if err != b"":
            raise RuntimeError(err.decode())
        return out.decode().strip()
    return __wrapper


def compare_files(path1, path2, algo="ssdeep"):
    if algo not in _HASH_FORMATS_REGEX:
        raise RuntimeError(f"'{algo}' is not a valid fuzzy hashing algorithm")
    if algo == "mrsh-v2":
        out = subprocess.check_output(["mrsh-v2", "-c", str(path1), str(path2)], stderr=subprocess.PIPE).decode()
        return int(out.strip().split("\n")[-1].split("|")[-1] or 0)  # 0-100
    elif algo == "sdhash":
        out = subprocess.check_output(["sdhash", "-g", str(path1), str(path2)], stderr=subprocess.PIPE).decode()
        return int(out.strip().split("\n")[-1].split("|")[-1] or 0)  # 0-100
    else:
        h = lambda p: p.fuzzy_hash if hasattr(p, "fuzzy_hash") else globals()[algo](p)
        return compare_fuzzy_hashes(h(path1), h(path2))


def compare_fuzzy_hashes(hash1, hash2):
    f1, f2 = _hash_format(h1 := hash1), _hash_format(h2 := hash2)
    if f1 != f2:
        raise RuntimeError(f"hash formats of '{string.shorten(h1, 30)}' ({f1}) and '{string.shorten(h2, 30)}' ({f2}) "
                           f"do not match")
    if f1 == "sdhash":
        tmp = TempPath(length=32)
        hf1, hf2 = tmp.tempfile(), tmp.tempfile()
        hf1.write_text(h1), hf2.write_text(h2)
        out = subprocess.check_output(["sdhash", "-c", str(hf1), str(hf2)], stderr=subprocess.PIPE).decode()
        tmp.remove()
        return int(out.strip().split("\n")[-1].split("|")[-1] or 0)  # 0-100
    elif f1 == "ssdeep":
        if (score := _FUZZY_LIB.fuzzy_compare(_new_cstr(h1.encode()), _new_cstr(h2.encode()))) < 0:
            raise RuntimeError(f"fuzzy_compare based on ssdeep failed ({score})")
        return score  # 0-100
    elif f1 == "tlsh":
        from math import exp
        from tlsh import diff
        dist = diff(h1, h2)
        #TODO: distance !!! 0 means identical, similarity decreases when score is increasing => requires calibration
        return dist
    raise RuntimeError(f"{f1} hashes comparison is not supported ; try 'compare_files' instead")


@_tool_wrapper
def mrsh_v2(path, **parameters):
    """ MRSH-v2 (Multi-Resolution Similarity Hashing version 2)
    - Variable-length output: proportional to file size and depending on input parameters
    - Format:                 [input filename]:[file size]:[number of bloom filters]:[block size]:
                               [serialized bloom filters (hex)]
    - Comparison:             similarity score (0-100)
    - Requirements:           any input size, but more effective with larger files
    """


def mvhash_b(path, **parameters):
    """ MVhash-B
    - Fixed-length output: typically 160 bits (SHA1), can be longer depending on hash algorithm used
    - Format:              [input filename]:[file size]:[hash]
    - Comparison:          Hamming distance (number of differing bits)
    - Requirements:        any input size

    Note: No implementation found for this one.
    """


@_tool_wrapper
def sdhash(path, **parameters):
    """ sdhash (Similarity Digest)
    - Variable-length output: depending on the input data’s entropy and content
    - Format:                 sdbf:[version]:[number of bloom filters]:[input filename]:[file size]:
                               [underlying hash algorithm for chunking]:[block size]:[number of features per block]:
                               [feature mask or bloom filter width/size]:[number of bits per bloom filter]:
                               [feature threshold]:[feature threshold]:[base64-encoded bloom filter(s)]
    - Comparison:             similarity score (0-100), NOT matching a percentage of commonality
    - Requirements:           any input size, but effectiveness increases with data size and entropy
    """


def ssdeep(path):
    """ ssdeep (SpamSum Deep)
    - Variable-length output: single line, typically 45-100 characters
    - Format:                 [blocksize]:[hash1]:[hash2]
    - Comparison:             similarity score (0-100)
    - Requirements:           any input size, but more effective with files >1KB
    """
    out = _FFI.new(f"char[{_FUZZY_MAX_RESULT}]")
    r = _FUZZY_LIB.fuzzy_hash_filename(_new_cstr(str(path).encode("utf-8")), out)
    if r != 0:
        raise RuntimeError(f"fuzzy_hash_filename with ssdeep failed ({r})")
    return _FFI.string(out).decode("ascii")


def tlsh(path):
    """ TLSH (Trend Micro Locality Sensitive Hash)
    - Fixed-length output: standard is 70 characters, can vary with options
    - Format:              upper-case hex
    - Comparison:          tlsh.diff provides a distance metric (lower means more similar)
    - Requirements:        input must be at least 256 bytes (by default) to compute a hash ; shorter inputs are rejected
    """
    from tlsh import hash as tlsh
    with open(str(path), 'rb') as f:
        content = f.read()
    return tlsh(content)

