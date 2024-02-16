# -*- coding: UTF-8 -*-
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
    lief.logging.enable() if config['lief_logging'] else lief.logging.disable()
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
        if self._parsed is not None and cls.__name__.upper() == self._parsed.format.__name__:
            return self
    
    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self, "_parsed") and hasattr(self._parsed, name):
                return getattr(self._parsed, name)
            raise
    
    def _get_builder(self):
        raise NotImplementedError  # to be implemented per binary format
    
    def build(self):
        p = str(self.path)
        builder = self._get_builder()
        #FIXME: fails with NSPack and LIEF 0.13.0 => fixed with 0.14.0 ?
        builder.build()
        builder.write(p)
        with open(p, 'ab') as f:
            f.write(bytes(self.overlay))
        return builder.get_build()
    
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
    def __init__(self, *keys, **kwargs):
        self.__keys = [k for k in keys if isinstance(k, str)]
        if len(self.__keys) == 0:
            raise ValueError("Empty build configuration dictionary")
        self.update(**kwargs)
    
    def __getitem__(self, name):
        if name not in self.__keys:
            raise KeyNotAllowedError(name)
        return super(BuildConfig, self).get(name, False)
    
    def __setitem__(self, name, value):
        if name not in self.__keys:
            raise KeyNotAllowedError(name)
        if not isinstance(value, bool):
            raise ValueError("Value shall be a boolean")
        super(BuildConfig, self).__setitem__(name, value)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

