# -*- coding: UTF-8 -*-
from tinyscript import functools, os
from tinyscript.helpers import Path

from ..__common__ import *


__all__ = ["get_section_class", "lief", "Binary", "BuildConfig"]


def __init_CS2CS_MODE():
    from capstone import CS_ARCH_ARM, CS_ARCH_ARM64, CS_ARCH_MIPS, CS_ARCH_PPC, CS_ARCH_X86, CS_MODE_ARM, \
                         CS_MODE_MIPS32, CS_MODE_MIPS64, CS_MODE_32, CS_MODE_64
    return {
        CS_ARCH_ARM:   {32: CS_MODE_ARM,    64: CS_MODE_ARM},
        CS_ARCH_ARM64: {32: CS_MODE_ARM,    64: CS_MODE_ARM},
        CS_ARCH_MIPS:  {32: CS_MODE_MIPS32, 64: CS_MODE_MIPS64},
        CS_ARCH_PPC:   {32: CS_MODE_32,     64: CS_MODE_64},
        CS_ARCH_X86:   {32: CS_MODE_32,     64: CS_MODE_64},
    }
lazy_load_object("_CS2CS_MODE", __init_CS2CS_MODE)


def __init_LIEF2CS_ARCH():
    from capstone import CS_ARCH_ARM, CS_ARCH_ARM64, CS_ARCH_MIPS, CS_ARCH_PPC, CS_ARCH_SPARC, CS_ARCH_SYSZ, \
                         CS_ARCH_X86, CS_ARCH_XCORE
    from lief import ARCHITECTURES as ARCH
    return {
        ARCH.ARM:   {32: CS_ARCH_ARM,   64: CS_ARCH_ARM64},
        ARCH.ARM64: {32: CS_ARCH_ARM,   64: CS_ARCH_ARM64},
        ARCH.INTEL: {32: CS_ARCH_X86,   64: CS_ARCH_X86},
        ARCH.MIPS:  {32: CS_ARCH_MIPS,  64: CS_ARCH_MIPS},
        ARCH.PPC:   {32: CS_ARCH_PPC,   64: CS_ARCH_PPC},
        ARCH.SPARC: {32: CS_ARCH_SPARC, 64: CS_ARCH_SPARC},
        ARCH.SYSZ:  {32: CS_ARCH_SYSZ,  64: CS_ARCH_SYSZ},
        ARCH.X86:   {32: CS_ARCH_X86,   64: CS_ARCH_X86},
        ARCH.XCORE: {32: CS_ARCH_XCORE, 64: CS_ARCH_XCORE},
    }
lazy_load_object("_LIEF2CS_ARCH", __init_LIEF2CS_ARCH)


def __init_lief(lief):
    errors = config['lief_errors']
    # define a decorator that handle LIEF errors setting
    def _handle_errors(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            if not errors:
                # capture the stderr messages from LIEF
                tmp_fd, null_fd = os.dup(2), os.open(os.devnull, os.O_RDWR)
                os.dup2(null_fd, 2)
            r = f(*args, **kwargs)
            if not errors:
                # restore stderr
                os.dup2(tmp_fd, 2)
                os.close(null_fd)
            return r
        return _wrapper
    # redefine parsing function to throw an error when the binary could not be parsed
    def _lief_parse(target, *args, **kwargs):
        target = Path(target, expand=True)
        if not target.exists():
            raise OSError("'%s' does not exist" % target)
        binary = _handle_errors(lief._parse)(str(target))
        if binary is None:
            raise OSError("'%s' has an unknown format")
        return binary
    lief._parse, lief.parse = lief.parse, _lief_parse
    # monkey-patch header-related classes ; this allows following calls (given that a parsed object from the Executable
    #  abstraction has this __getitem__ method too):
    #   exe.parsed['header']['signature']
    #   exe.parsed['optional_header']['file_alignment']
    # this is for use in YAML configurations (parameter 'result')
    def __getitem__(self, name):
        if not name.startswith("_"):
            return getattr(self, name)
        raise KeyNotAllowedError(name)
    lief._lief.PE.Header.__getitem__ = __getitem__
    lief._lief.PE.OptionalHeader.__getitem__ = __getitem__
    return lief
lazy_load_module("lief", postload=__init_lief)


class Binary(AbstractParsedExecutable):
    def __new__(cls, path, *args, **kwargs):
        self = super().__new__(cls)
        self._parsed = lief.parse(str(path))
        if self._parsed is not None and cls.__name__.upper() == self._parsed.format.name:
            return self
    
    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self, "_parsed") and hasattr(self._parsed, name):
                return getattr(self._parsed, name)
            raise
    
    def build(self):
        builder = self._get_builder()
        builder.build()
        builder.write(self.name)
        with open(self.name, 'ab') as f:
            f.write(bytes(self.overlay))
    
    def disassemble(self, offset=None, n=32, mnemonic=False):
        from capstone import Cs, CS_MODE_LITTLE_ENDIAN as CS_MODE_LE
        offset, idc = offset or self.entrypoint, [32, 64][self.path.format[-2:] == "64"]
        arch = _LIEF2CS_ARCH[self.architecture][idc]
        disassembler = Cs(arch, _CS2CS_MODE.get(arch, {}).get(idc, CS_MODE_LE))
        r = []
        for i in disassembler.disasm(bytes(self.get_content_from_virtual_address(offset, n*8)), offset):
            if len(r) >= n:
                break
            r.append(i.mnemonic if mnemonic else i.id)
        while len(r) < n:
            r.append(-1)
        return r


class BuildConfig(dict):
    def toggle(self, **kwargs):
        for name, boolean in kwargs.items():
            self[name] = self.get(name, True) & boolean

