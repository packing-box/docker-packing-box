# -*- coding: UTF-8 -*-
from tinyscript import hashlib, logging, os, re, string
from tinyscript.helpers import execute_and_log, set_exception, Path, TempPath

from .fuzzhash import *
from .rendering import progress_bar


__all__ = ["filter_archive", "get_archive_class", "is_archive", "read_archive", "write_archive"]


set_exception("BadFileFormat", "OSError")


@logging.bindLogger
def filter_archive(path, output, filter_func=None, similarity_algorithm=None, similarity_threshold=None,
                   keep_files=False, no_ext=False):
    from .fuzzhash import __all__ as _fhash
    if similarity_algorithm not in \
       (_fh := list(map(lambda x: x.replace("_", "-"), filter(lambda x: not x.startswith("compare_"), _fhash)))):
        raise ValueError(f"'{similarity_algorithm}' is not a valid algorithm ; should be one of {', '.join(_fh)}")
    tp, out, hashes, cnt = TempPath(prefix="output_", length=16), Path(output), {}, 0
    with progress_bar(unit="files", target=str(out)) as progress:
        pbar = progress.add_task("", total=None)
        for fp in read_archive(path, filter_func=filter_func, keep_files=keep_files, no_ext=no_ext):
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
                            logger.debug(f"Hash comparison failed\n{h1}\n{h2}")
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
    write_archive(out, tp, archive_class=arch_cls, logger=logger)
    if logger:
        logger.info(f"Filtered {cnt} files out of {path}")


@logging.bindLogger
def get_archive_class(path, origin=None):
    for cls in _ARCHIVE_CLASSES:
        try:
            p = cls(path, test=True)
            if logger:
                logger.info(f"Found {cls.__name__[:-7]}: {origin or p}")
            return cls
        except BadFileFormat:
            pass
    next(GenericArchive(path).list())
    return GenericArchive


def is_archive(path):
    try:
        get_archive_class(path)
        return True
    except BadFileFormat:
        return False


@logging.bindLogger
def read_archive(path, destination=None, filter_func=None, path_cls=None, origin=None, keep_files=False, no_ext=False,
                 recurse=True):
    cnt = 0
    with get_archive_class(path, origin)(path, destination=destination, keep_files=keep_files, no_ext=no_ext) as p:
        origin = origin or Path(f"[{p.filename}:{hashlib.sha256_file(p)}]")
        # walk files from the target path
        for fp in ([p._dst] if p._dst.is_file() else p._dst.walk()):
            if not fp.is_file():
                continue
            if path_cls is not None:
                fp = path_cls(fp)
            fp._dst = p._dst
            fp.origin = origin
            # try to recurse archives if relevant
            try:
                if recurse:
                    for sfp in read_archive(fp, None, filter_func, path_cls, origin.joinpath(fp.relative_to(fp._dst)),
                                            keep_files, no_ext):
                        sfp._parent = fp.relative_to(fp._dst)
                        yield sfp
                    continue  # do not yield the file if it was itself an archive
            except BadFileFormat:
                pass
            # otherwise, if this path is filtered, stop here
            if callable(filter_func) and not filter_func(fp):
                continue
            # if not an archive or no recursion needed, just output the file path
            if logger:
                logger.debug(origin.joinpath(fp.relative_to(p._dst)))
            yield fp
            cnt += 1
    if logger and cnt > 0:
        logger.info(f"Selected {cnt} files from {origin}")


@logging.bindLogger
def write_archive(path, source=None, **kwargs):
    out, arch_cls = Path(path), kwargs.get("archive_class")
    # if the target is an empty folder, just do nothing
    if out.is_dir() and sum(1 for _ in out.listdir()) == 0:
        logger.warning("Nothing to archive")
        return
    # if the archive class is not specified, just base the archive type on the provided extension
    if arch_cls is None:
        for cls in _ARCHIVE_CLASSES:
            if getattr(cls, "base_extension", None) == out.extension:
                arch_cls = cls
                break
    if arch_cls is None:
        logger.warning("Unsupported target archive type")
        return
    with arch_cls(out, source=Path(source or out.dirname.joinpath(out.stem)), **kwargs) as arch:
        arch.write()


class Archive(Path):
    def __new__(cls, *parts, **kwargs):
        p = super(Archive, cls).__new__(cls, *parts, **kwargs)
        p.mode = kwargs.get('mode', 'wr'[(tst := kwargs.get('test', False)) or p.exists()])
        p.__keep = kwargs.get('keep_files', False)
        if p.mode in 'ar':
            if p.mode == 'r' and (src := kwargs.get('source')) is not None:
                raise FileExistsError(f"File '{p}' already exists")
            with p.open('rb') as f:
                sig = f.read(0x9010)  # ISO has signature pattern up to 0x9001 bytes ("CD001"), hence choosing 0x9010
            if cls is not GenericArchive and (not cls.signature(sig) or \
               (not kwargs.get('no_ext', False) and (e := p.extension) != getattr(cls, "base_extension", e))):
                raise BadFileFormat(f"Not a {cls.__name__[:-7]} file")
            if not tst:
                if (p2 := kwargs.get('destination')):
                    p._dst = Path(p2)
                else:
                    p._dst = TempPath(prefix="archive_", length=16)
                    p._dst._created = True
            if p.mode == 'r':
                return p
        p._src = Path(kwargs.get('source', "."))
        if not p._src.exists():
            raise OSError(f"Folder {p._src} does not exist")
        return p
    
    def __enter__(self):
        if self.mode == 'r':
            try:
                self._dst.mkdir()
                self._dst._created = True
            except FileExistsError:
                self._dst._created = getattr(self._dst, "_created", False)
            if getattr(self, "extract", getattr(self, "mount", lambda: 0))() == 0:
                raise NotImplementedError
        return self
    
    def __exit__(self, *args, **kwargs):
        if self.mode == 'r':
            getattr(self, "unmount", lambda: 0)()
            if (p := self._dst)._created and (not self.__keep or sum(1 for _ in p.listdir()) == 0):
                p.remove()
    
    def write(self):
        raise NotImplementedError


class GenericArchive(Archive):
    def extract(self):
        execute_and_log(f"unar \"{self}\" -o \"{self._dst}\"",
                        silent=["invalid (second|minute|hour|day|month|year) given"])
    
    def list(self):
        out, _, retc = execute_and_log(f"lsar \"{self}\"", silent=["invalid (second|minute|hour|day|month|year) given",
                                                                   "Couldn't recognize the archive format."])
        if retc > 0:
            raise BadFileFormat("Unsuppored archive type")
        for line in out.decode().splitlines():
            if line.lstrip(".").rstrip("/") == "":
                continue
            yield Path(line)


class CABArchive(Archive):
    base_extension = ".cab"
    signature = lambda bytes: bytes[:4] == b"MSCF" and bytes[24:26] == b"\x03\x01"  # magic bytes and fixed version
    
    def extract(self):
        execute_and_log(f"gcab -x \"{self}\" -nC \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"gcab -c \"{self}\" \"{self._src}\"/*")


class DEBArchive(Archive):
    base_extension = ".deb"
    signature = lambda bytes: bytes[:8] == b"!<arch>\n"
    
    def extract(self):
        execute_and_log(f"dpkg-deb -R \"{self}\" \"{self._dst}\"")
    
    def write(self):
        execute_and_log(f"dpkg-deb -b \"{self._src}\" \"{self.stem}.deb\"")


class GZArchive(Archive):
    signature = lambda bytes: bytes[:2] == b"\x1f\x8b" or bytes[:4] == b"\x28\xb5\x2d\xfd"  # respectively GZ and ZStd
    tool = "gzip"
    
    def extract(self):
        if self.extension == ".gz":
            self._dst = self.dirname.joinpath(self.stem)  # remove the ".gz" extension
            self._dst._created = True
            execute_and_log(f"{self.tool} -d \"{self}\"")
        else:
            execute_and_log(f"tar xf \"{self}\" --directory=\"{self._dst}\"")
    
    def write(self):
        tgt = f"--directory=\"{self._src}\" ." if self._src.is_dir() else f"\"{self._src}\""
        execute_and_log(f"tar {['', 'r'][self.mode == 'a']}cf \"{self.stem}.tar\" {tgt}")
        execute_and_log(f"{self.tool} \"{self.stem}.tar\"")


class BZ2Archive(GZArchive):
    base_extension = ".tar.gz"
    signature = lambda bytes: bytes[:3] == b"BZh"
    tool = "bzip2"


class XZArchive(GZArchive):
    base_extension = ".tar.xz"
    signature = lambda bytes: bytes[:6] == b"\xfd7zXZ\x00"
    tool = "xz"


class ISOArchive(Archive):
    base_extension = ".iso"
    signature = lambda bytes: all(bytes[n:n+5] == b"CD001" for n in [0x8001, 0x8801])  # 0x9001
    
    @logging.bindLogger
    def extract(self):
        execute_and_log(f"unar \"{self}\" -o \"{self._dst}\"",
                        silent=["invalid (second|minute|hour|day|month|year) given"])
        if (p := self._dst.joinpath("README.TXT")).exists():
            # special case of UDF ISO image
            if "This disc contains a \"UDF\" file system and requires an operating system that supports the ISO-13346" \
               " \"UDF\" file system specification." in p.read_text().replace("\n", " "):
                from pycdlib import PyCdlib
                from tinyscript.helpers import human_readable_size
                self.logger.debug(f"This is not an ISO 9660, parsing as UDF instead")
                p.remove()
                self.logger.debug(f"{self}: UDF")
                iso = PyCdlib()
                iso.open(self)
                try:
                    for udf_path, dirs, files in iso.walk(udf_path="/"):
                        host_dir = self._dst.joinpath(udf_path.lstrip("/"))
                        host_dir.mkdir(exist_ok=True)
                        for name in files:
                            udf_file = Path(udf_path).joinpath(name)
                            iso.get_file_from_iso(f := host_dir.joinpath(name.split(';')[0]), udf_path=str(udf_file))
                            self.logger.debug(f"  {udf_file}  ({human_readable_size(f.size)})... OK.")
                except Exception as e:
                    self.logger.error(e)
                iso.close()
        elif (p := self._dst.joinpath(self.stem)).exists() and sum(1 for _ in self._dst.listdir()) == 1:
            p.rename(dst := TempPath(prefix="archive_", length=16))
            self._dst.remove()
            self._dst = dst
            self._dst._created = True
        
    def write(self):
        execute_and_log(f"genisoimage -U -R -o \"{self}\" \"{self._src}\"", silent=["-input-charset not specified"])


class LZHArchive(Archive):
    signature = lambda bytes: re.match(rb"\-lh\d\-", bytes[2:7])
    
    def extract(self):
        execute_and_log(f"jlha -xw=\"{self._dst}\" \"{self}\"")
    
    def write(self):
        execute_and_log(f"jlha a \"{self}\" \"{self._src}\"")


class RPMArchive(Archive):
    signature = lambda bytes: bytes[:4] == b"\xed\xab\xee\xdb"
    
    def extract(self):
        execute_and_log(f"rpm2archive \"{self}\"")
        execute_and_log(f"tar xf \"{self}.tgz\" --directory=\"{self._dst}\"")


class WIMArchive(Archive):
    base_extension = ".wim"
    signature = lambda bytes: bytes[:8] == b"MSWIM\0\0\0" or bytes[:8] == b"WLPWM\0\0\0"
    
    def extract(self):
        execute_and_log(f"wimlib-imagex extract \"{self}\" 1 --dest-dir \"{self._dst}\"", silent=["[WARNING] "])
    
    def write(self):
        execute_and_log(f"wimlib-imagex capture \"{self._src}\" \"{self}\"")


class ZIPArchive(Archive):
    base_extension = ".zip"
    signature = lambda bytes: bytes[:2] == b"PK" and bytes[2:4] in [b"\x03\x04", b"\x05\x06", b"\x07\x08"]
    
    def extract(self):
        execute_and_log(f"unzip -o \"{self}\" -d \"{self._dst}\"")
    
    def write(self):
        dst, cwd = self.absolute(), os.getcwd()
        os.chdir(self._src.parent if self._src.is_file() else self._src)
        execute_and_log(f"zip -r -9 \"{dst}\" " + [".", f"\"{self._src.basename}\""][self._src.is_file()])
        os.chdir(cwd)


class ZSTDArchive(Archive):
    base_extension = ".zst"
    signature = lambda bytes: lambda bytes: bytes[:4] == b"\x28\xb5\x2f\xfd"
    
    def extract(self):
        execute_and_log(f"unzstd \"{self}\" --output-dir-flat \"{self._dst}\"")
        p = next(self._dst.listdir())
        if p.extension != ".tar":
            p = Path(p.rename(f"{p}.tar"))
        execute_and_log(f"tar xf {p} -C \"{self._dst}\"")
        p.remove()


_ARCHIVE_CLASSES = [cls for cls in globals().values() if hasattr(cls, "__base__") and \
                    cls.__base__.__name__.endswith("Archive") and cls.__name__ != "GenericArchive"]
_WRITEABLE_ARCHIVE_EXTS = sorted([cls.base_extension for cls in _ARCHIVE_CLASSES if hasattr(cls, "base_extension")])
__all__.extend([cls.__name__ for cls in _ARCHIVE_CLASSES])

