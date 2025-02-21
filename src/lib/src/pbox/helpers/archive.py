# -*- coding: UTF-8 -*-
from tinyscript import re
from tinyscript.helpers import execute_and_log, set_exception, Path, TempPath


__all__ = ["read_archive"]


set_exception("BadFileFormat", "OSError")


def read_archive(path, destination=None, filter_func=None, logger=None):
    p = None
    for cls in globals().values():
        if hasattr(cls, "__base__") and cls.__base__ is Archive:
            try:
                p = cls(path, test=True)
                if logger:
                    logger.debug(f"Found {cls.__name__[:-7]}: {p}")
                break
            except BadFileFormat:
                pass
    if p is None:
        raise BadFileFormat("Unsuppored archive type")
    with cls(path, destination=destination) as p:
        for fp in p._dest.walk():
            if logger:
                logger.debug(fp)
            if filter_func(fp):
                yield fp
                continue
            if not fp.is_file():
                continue
            try:
                for sfp in read_archive(fp, None, filter_func, logger):
                    processed = True
                    yield sfp
            except BadFileFormat:
                pass


class Archive(Path):
    def __new__(cls, *parts, **kwargs):
        p = super(Archive, cls).__new__(cls, *parts, **kwargs)
        with p.open('rb') as f:
            sig = f.read(0x9010)
        if not cls.signature(sig):
            raise BadFileFormat(f"Not a {cls.__name__[:-7]} file")
        if not kwargs.get('test', False):
            if (p2 := kwargs.get('destination')):
                p._dest = Path(p2)
            else:
                p._dest = TempPath(prefix="archive_", length=16)
                p._dest._created = True
        return p
    
    def __enter__(self):
        if not hasattr(self._dest, "_created"):
            try:
                self._dest.mkdir()
                self._dest._created = True
            except FileExistsError:
                self._dest._created = False
        getattr(self, "extract", getattr(self, "mount", lambda: 0))()
        return self
    
    def __exit__(self, *args, **kwargs):
        getattr(self, "unmount", lambda: 0)()
        if (p := self._dest)._created:
            p.remove()


class CABArchive(Archive):
    signature = lambda bytes: bytes[:4] == b"MSCF" and bytes[24:26] == b"\x03\x01"  # magic bytes and fixed version
    
    def extract(self):
        execute_and_log(f"cabextract -d \"{self._dest}\" \"{self}\"")


class ISOArchive(Archive):
    signature = lambda bytes: all(bytes[n:n+5] == b"CD001" for n in [0x8001, 0x8801, 0x9001])
    
    def extract(self):
        execute_and_log(f"7z x \"{self}\" -o\"{self._dest}\"")


class WIMArchive(Archive):
    signature = lambda bytes: bytes[:8] == b"MSWIM\0\0\0" or bytes[:8] == b"WLPWM\0\0\0"
    
    def extract(self):
        execute_and_log(f"wimlib-imagex extract \"{self}\" 1 --dest-dir \"{self._dest}\"")


class ZIPArchive(Archive):
    signature = lambda bytes: bytes[:2] == b"PK" and bytes[2:4] in [b"\x03\x04", b"\x05\x06", b"\x07\x08"]
    
    def extract(self):
        execute_and_log(f"unzip \"{self}\" -d \"{self._dest}\"")

