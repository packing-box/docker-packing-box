# -*- coding: UTF-8 -*-
""" Line-based editing of the YAML definitions (detectors.yml, features.yml, ...).

The definition files are hand-written: entries are top-level keys whose properties are indented by two spaces, with
 comments and blank lines that matter to their maintainers. A YAML round trip (load + dump) would lose these, so this
 module edits the lines of a single entry instead, leaving the rest of the file untouched.
"""
import re
from pathlib import Path


__all__ = ["definition_files", "find_entry", "get_block", "remove_property", "set_property"]

_TOP = re.compile(r"^([^\s#][^:]*?)\s*:\s*(?:#.*)?$")
_PROP = re.compile(r"^  ([^\s#][^:]*?)\s*:(.*)$")


def definition_files(path):
    """ YAML files holding the definitions at path: the file itself, or every .yml file of the folder (recursively),
         as pbox's load_yaml_config does. """
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.rglob("*.yml") if f.stem != "defaults")
    return [p]


def _bounds(lines, key):
    """ First and last+1 line indices of the block of the top-level 'key', or None. Trailing blank and comment lines
         are left out of the block (they usually introduce the next entry). """
    start = None
    for i, l in enumerate(lines):
        m = _TOP.match(l.rstrip("\n"))
        if start is None:
            if m and m.group(1).strip("'\"") == key:
                start = i
        elif m or (l.strip() and not l[0].isspace() and not l.lstrip().startswith("#")):
            end = i
            break
    else:
        if start is None:
            return None
        end = len(lines)
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        end -= 1
    return start, end


def find_entry(path, key):
    """ File (among the definition files at path) that defines the top-level 'key', or None. """
    for f in definition_files(path):
        try:
            if _bounds(f.read_text(encoding="utf-8").splitlines(True), key):
                return f
        except (OSError, UnicodeDecodeError):
            continue


def get_block(path, key):
    """ YAML text of the entry 'key' (searched among the definition files at path), or None. """
    if (f := find_entry(path, key)) is None:
        return
    lines = f.read_text(encoding="utf-8").splitlines(True)
    s, e = _bounds(lines, key)
    return "".join(lines[s:e]).rstrip("\n")


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value)
    # quote values that YAML would not read back as the same string
    if s == "" or re.search(r"[:#\[\]{},&*!|>'\"%@`]", s) or s.lower() in ("true", "false", "yes", "no", "null", "~"):
        return "'" + s.replace("'", "''") + "'"
    return s


def set_property(path, key, prop, value):
    """ Set the scalar property 'prop' of the entry 'key' to 'value', in place. The property is replaced where it is
         (keeping an inline comment), or inserted among the properties of the entry at its alphabetical position (the
         files keep them sorted). Returns the modified file.

        Raises KeyError if no definition file holds 'key', ValueError if 'prop' holds a nested value (list or
         mapping) that cannot be replaced by a scalar. """
    if (f := find_entry(path, key)) is None:
        raise KeyError(f"'{key}' is not defined in {path}")
    text = f.read_text(encoding="utf-8")
    lines = text.splitlines(True)
    s, e = _bounds(lines, key)
    nl = "\r\n" if lines[s].endswith("\r\n") else "\n"
    new = f"  {prop}: {_fmt(value)}"
    insert_at = None
    for i in range(s + 1, e):
        m = _PROP.match(lines[i].rstrip("\r\n"))
        if not m:
            continue
        name, rest = m.group(1).strip("'\""), m.group(2)
        if name == prop:
            if rest.strip() == "" or rest.strip().startswith("#"):
                raise ValueError(f"'{prop}' of '{key}' is not a scalar")
            comment = re.search(r"\s+#.*$", rest)
            # only keep the comment if the value is not quoted (a '#' inside quotes is not a comment)
            keep = comment and not re.match(r"\s*['\"]", rest)
            lines[i] = new + (comment.group(0) if keep else "") + nl
            break
        if insert_at is None and name > prop:
            insert_at = i
    else:
        if insert_at is None:
            # after the last non-blank line of the block
            insert_at = e
            while insert_at > s + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
        if insert_at > 0 and not lines[insert_at - 1].endswith(("\n", "\r")):
            lines[insert_at - 1] += nl
        lines.insert(insert_at, new + nl)
    f.write_text("".join(lines), encoding="utf-8")
    return f


def remove_property(path, key, prop):
    """ Remove the scalar property 'prop' of the entry 'key' (so that the defaults apply again), in place. Returns the
         modified file, or None if the entry had no such property.
        
        Raises KeyError if no definition file holds 'key', ValueError if 'prop' holds a nested value. """
    if (f := find_entry(path, key)) is None:
        raise KeyError(f"'{key}' is not defined in {path}")
    lines = f.read_text(encoding="utf-8").splitlines(True)
    s, e = _bounds(lines, key)
    for i in range(s + 1, e):
        m = _PROP.match(lines[i].rstrip("\r\n"))
        if m and m.group(1).strip("'\"") == prop:
            if m.group(2).strip() == "" or m.group(2).strip().startswith("#"):
                raise ValueError(f"'{prop}' of '{key}' is not a scalar")
            del lines[i]
            f.write_text("".join(lines), encoding="utf-8")
            return f
