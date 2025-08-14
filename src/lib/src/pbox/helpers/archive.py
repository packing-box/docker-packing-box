# -*- coding: UTF-8 -*-
from tinyscript import hashlib, re
from tinyscript.helpers import execute_and_log, set_exception, Path, TempPath

from .fuzzhash import *
from .rendering import progress_bar


__all__ = ["filter_archive", "get_archive_class", "is_archive", "read_archive"]


set_exception("BadFileFormat", "OSError")


def filter_archive(path, output, filter_func=None, similarity_algorithm=None, similarity_threshold=None, logger=None,
                   keep_files=False):
    from .fuzzhash import __all__ as _fhash
    if similarity_algorithm not in \
       (_fh := list(map(lambda x: x.replace("_", "-"), filter(lambda x: not x.startswith("compare_"), _fhash)))):
        raise ValueError(f"'{similarity_algorithm}' is not a valid algorithm ; should be one of {', '.join(_fh)}")
    tp, out, hashes, cnt = TempPath(prefix="output_", length=16), Path(output), {}, 0
    with progress_bar(target=str(out)) as progress:
        pbar = progress.add_task("", total=None)
        for fp in read_archive(path, filter_func=filter_func, logger=logger, keep_files=keep_files):
            progress.update(pbar, advance=1)
            rp = fp.relative_to(fp._dst)
            # similarity_threshold=None means do not filter anything
            if similarity_threshold is not None:
                h1, discard = globals()[similarity_algorithm.replace("-", "_")](fp), None
                for h2, f2 in hashes.items():
                    try:
                        score = compare_fuzzy_hashes(h1, h2)
                    except ValueError:
                        score = compare_files(fp, f2, similarity_algorithm)
                    except RuntimeError:
                        if logger:
                            logger.debug(f"Hash comparison failed: {h1} - {h2}")
                        continue
                    if score >= similarity_threshold:
                        discard = f2
                        break
                hashes[h1] = str(fp)
                if discard is not None:
                    if logger:
                        logger.debug(f"Discarded \"{rp}\" because of similarity ({score}%) with \"{f2}\"")
                    continue
            np = tp.joinpath(getattr(fp, "_parent", Path("."))).joinpath(*rp.parts)
            np.dirname.mkdir(parents=True, exist_ok=True)
            fp.copy(np)
            cnt += 1
    arch_cls, exts = None, []
    for cls in _ARCHIVE_CLASSES:
        if out.extension.split(".")[-1].lower() == (ext := cls.__name__[:-7].lower()):
            arch_cls = cls
        exts.append(ext)
    if arch_cls is None:
        raise ValueError(f"Output's extension shall be one of: {'|'.join(exts)}")
    with arch_cls(out, source=tp) as arch:
        arch.write()
    if logger:
        logger.info(f"Filtered {cnt} files out of {path}")


def get_archive_class(path, logger=None):
    for cls in _ARCHIVE_CLASSES:
        try:
            p = cls(path, test=True)
            if logger:
                logger.info(f"Found {cls.__name__[:-7]}: {p}")
            return cls
        except BadFileFormat:
            pass
    raise BadFileFormat("Unsuppored archive type")


def is_archive(path, logger=None):
    try:
        get_archive_class(path, logger)
        return True
    except BadFileFormat:
        return False


def read_archive(path, destination=None, filter_func=None, logger=None, path_cls=None, origin=None, keep_files=False):
    cnt = 0
    with get_archive_class(path, logger)(path, destination=destination, keep_files=keep_files) as p:
        origin = origin or Path(f"[{p.filename}:{hashlib.sha256_file(p)}]")
        for fp in p._dst.walk():
            if not fp.is_file():
                continue
            if path_cls is not None:
                fp = path_cls(fp)
            fp._dst = p._dst
            fp.origin = origin
            if callable(filter_func) and filter_func(fp):
                if logger:
                    logger.debug(origin.joinpath(fp.relative_to(p._dst)))
                yield fp
                cnt += 1
                continue
            try:
                for sfp in read_archive(fp, None, filter_func, logger, path_cls,
                                        origin.joinpath(fp.relative_to(fp._dst))):
                    sfp._parent = fp.relative_to(fp._dst)
                    yield sfp
            except BadFileFormat:
                if filter_func is None:
                    if logger:
                        logger.debug(origin.joinpath(fp.relative_to(p._dst)))
                    yield fp
                    cnt += 1
    if logger:
        logger.info(f"Selected {cnt} files from {origin}")


class Archive(Path):
    def __new__(cls, *parts, **kwargs):
        p = super(Archive, cls).__new__(cls, *parts, **kwargs)
        p.mode = 'wr'[(tst := kwargs.get('test', False)) or p.exists()]
        p.__keep = kwargs.get('keep_files', False)
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
            if (p := self._dst)._created and not self.__keep:
                p.remove()
    
    def write(self):
        raise NotImplementedError


class CABArchive(Archive):
    signature = lambda bytes: bytes[:4] == b"MSCF" and bytes[24:26] == b"\x03\x01"  # magic bytes and fixed version
    
    def extract(self):
        execute_and_log(f"gcab -x \"{self}\" -nC \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"gcab -c \"{self}\" \"{self._src}\"/*")


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


class ISOArchive(Archive):
    signature = lambda bytes: all(bytes[n:n+5] == b"CD001" for n in [0x8001, 0x8801])  # 0x9001
    
    def extract(self):
        execute_and_log(f"7z x \"{self}\" -o\"{self._dst}\"")
        # IMPORTANT NOTE: ISO cannot be mounted within a Docker box unless this gets run with full administrative
        #                  privileges, hence using extraction with 7zip instead, which may take a while
    
    def write(self):
        execute_and_log(f"genisoimage -o \"{self}\" \"{self._src}\"", silent=["-input-charset not specified"])


class WIMArchive(Archive):
    signature = lambda bytes: bytes[:8] == b"MSWIM\0\0\0" or bytes[:8] == b"WLPWM\0\0\0"
    
    def extract(self):
        execute_and_log(f"wimlib-imagex extract \"{self}\" 1 --dest-dir \"{self._dst}\"", silent=["[WARNING] "])
    
    def write(self):
        execute_and_log(f"wimlib-imagex capture \"{self._src}\" \"{self}\"")


class ZIPArchive(Archive):
    signature = lambda bytes: bytes[:2] == b"PK" and bytes[2:4] in [b"\x03\x04", b"\x05\x06", b"\x07\x08"]
    
    def extract(self):
        execute_and_log(f"unzip \"{self}\" -d \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"zip -r 9 \"{self}\" \"{self._src}\"")


_ARCHIVE_CLASSES = [cls for cls in globals().values() if hasattr(cls, "__base__") and \
                                                         cls.__base__.__name__.endswith("Archive")]
__all__.extend([cls.__name__ for cls in _ARCHIVE_CLASSES])

