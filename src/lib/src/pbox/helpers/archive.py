# -*- coding: UTF-8 -*-
from tinyscript import re
from tinyscript.helpers import execute_and_log, set_exception, Path, TempPath


__all__ = ["filter_archive", "read_archive"]


set_exception("BadFileFormat", "OSError")


def filter_archive(path, output, filter_func=None, logger=None):
    tp, out = TempPath(prefix="output_", length=16), Path(output)
    for fp in read_archive(path, filter_func=filter_func, logger=logger):
        np = tp.joinpath(getattr(fp, "_parent", Path("."))).joinpath(*fp.relative_to(fp._dst).parts)
        np.dirname.mkdir(parents=True, exist_ok=True)
        fp.copy(np)
    arch_cls, exts = None, []
    for cls in _ARCHIVE_CLASSES:
        if out.extension.lower() == (ext := f".{cls.__name__[:-7].lower()}"):
            arch_cls = cls
        exts.append(ext)
    if arch_cls is None:
        raise ValueError(f"Output's extension shall be one of: {'|'.join(exts)}")
    with arch_cls(out, source=tp) as arch:
        arch.write()


def read_archive(path, destination=None, filter_func=None, logger=None):
    p = None
    for cls in _ARCHIVE_CLASSES:
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
        for fp in p._dst.walk():
            if logger:
                logger.debug(fp)
            if not fp.is_file():
                continue
            fp._dst = p._dst
            if callable(filter_func) and filter_func(fp):
                yield fp
                continue
            if filter_func is None:
                yield fp
            try:
                for sfp in read_archive(fp, None, filter_func, logger):
                    sfp._parent = fp.relative_to(fp._dst)
                    yield sfp
            except BadFileFormat:
                pass


class Archive(Path):
    def __new__(cls, *parts, **kwargs):
        p = super(Archive, cls).__new__(cls, *parts, **kwargs)
        p.mode = 'wr'[(tst := kwargs.get('test', False)) or p.exists()]
        if p.mode == 'r':
            if (src := kwargs.get('source')) is not None:
                raise FileExistsError(f"Archive '{p}' already exists")
            with p.open('rb') as f:
                sig = f.read(0x9010)
            if not cls.signature(sig):
                raise BadFileFormat(f"Not a {cls.__name__[:-7]} file")
            if not tst:
                if (p2 := kwargs.get('destination')):
                    p._dst = Path(p2)
                else:
                    p._dst = TempPath(prefix="archive_", length=16)
                    p._dst._created = True
        else:
            p._src = Path(kwargs.get('source', "."))
            if not p._src.exists():
                raise OSError(f"Folder {p._src} does not exist")
        return p
    
    def __enter__(self):
        if self.mode == 'r':
            if not hasattr(self._dst, "_created"):
                try:
                    self._dst.mkdir()
                    self._dst._created = True
                except FileExistsError:
                    self._dst._created = False
            getattr(self, "extract", getattr(self, "mount", lambda: 0))()
        return self
    
    def __exit__(self, *args, **kwargs):
        if self.mode == 'r':
            getattr(self, "unmount", lambda: 0)()
            if (p := self._dst)._created:
                p.remove()
    
    def write(self):
        raise NotImplementedError


class CABArchive(Archive):
    signature = lambda bytes: bytes[:4] == b"MSCF" and bytes[24:26] == b"\x03\x01"  # magic bytes and fixed version
    
    def extract(self):
        execute_and_log(f"gcab -x \"{self}\" -nC \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"gcab -c \"{self}\" \"{self._src}\"/*")


class ISOArchive(Archive):
    signature = lambda bytes: all(bytes[n:n+5] == b"CD001" for n in [0x8001, 0x8801, 0x9001])
    
    def extract(self):
        execute_and_log(f"7z x \"{self}\" -o\"{self._dst}\"")
        # IMPORTANT NOTE: ISO cannot be mounted within a Docker box unless this gets run with full administrative
        #                  privileges, hence using extraction with 7zip instead, which may take a while
    
    def write(self):
        execute_and_log(f"genisoimage -o \"{self}\" \"{self._src}\"", silent=["-input-charset not specified"])


class GZArchive(Archive):
    signature = lambda bytes: bytes[:2] == b"\x1f\x8b"
    tool = "gzip"
    
    def extract(self):
        self._dst.mkdir(exist_ok=True)
        execute_and_log(f"tar xf \"{self}\" --directory=\"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"tar cf \"{self.stem}.tar\" --directory=\"{self._src}\" .")
        execute_and_log(f"{self.tool} \"{self.stem}.tar\"")


class BZ2Archive(GZArchive):
    signature = lambda bytes: bytes[:3] == b"BZh"
    tool = "bzip2"


class XZArchive(GZArchive):
    signature = lambda bytes: bytes[:6] == b"\xfd7zXZ\x00"
    tool = "xz"


class WIMArchive(Archive):
    signature = lambda bytes: bytes[:8] == b"MSWIM\0\0\0" or bytes[:8] == b"WLPWM\0\0\0"
    
    def extract(self):
        execute_and_log(f"wimlib-imagex extract \"{self}\" 1 --dest-dir \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"wimlib-imagex capture \"{self._src}\" \"{self}\"")


class ZIPArchive(Archive):
    signature = lambda bytes: bytes[:2] == b"PK" and bytes[2:4] in [b"\x03\x04", b"\x05\x06", b"\x07\x08"]
    
    def extract(self):
        execute_and_log(f"unzip \"{self}\" -d \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"zip -r 9 \"{self}\" \"{self._src}\"")


_ARCHIVE_CLASSES = [cls for cls in globals().values() if hasattr(cls, "__base__") and cls.__base__ is Archive]
__all__.extend([cls.__name__ for cls in _ARCHIVE_CLASSES])

