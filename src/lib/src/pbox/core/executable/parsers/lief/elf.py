# -*- coding: UTF-8 -*-
from .__common__ import *


__all__ = ["ELF"]


_FLAGS = {
    'ALLOC':            "A",
    'COMPRESSED':       "C",
    'EXCLUDE':          "E",
    'GROUP':            "G",
    'INFO_LINK':        "I",
    'LINK_ORDER':       "L",
    'MERGE':            "M",
    'NONE':             "N",
    'OS_NONCONFORMING': "O",
    'STRINGS':          "S",
    'TLS':              "T",
    'WRITE':            "W",
    'EXECINSTR':        "X",
}
_FLAG_RENAME = {
    'ALLOC':     "allocated",
    'EXECINSTR': "executable",
    'WRITE':     "writable",
}
_SYMBOL_PENALTIES = {
    .01: ["__gmon_start__", "__cxa_atexit", "__register_frame", "__stack_chk_guard", "__dso_handle", "sigaltstack",
          "sigsuspend", "rt_sigaction", "kill", "madvise"],
    .02: ["__cxa_begin_catch", "__cxa_allocate_exception", "__gcc_personality_v0", "__tls_get_addr",
          "__libc_thread_freeres", "__gxx_personality_v0", "__sigsetjmp", "__cxa_throw", "munmap", "brk", "mlock",
          "shmget", "tgkill", "tkill", "sigaction", "pthread_create", "pthread_kill", "sched_setaffinity",
          "sched_getaffinity", "personality"],
    .03: ["gettid", "pivot_root", "openat2", "memfd_create", "vfork", "execveat", "getauxval", "dlclose", "backtrace",
          "backtrace_symbols"],
    .04: ["prctl", "fanotify_init", "landlock_add_rule", "mprotect", "arch_prctl", "dl_iterate_phdr", "dlsym"],
    .05: ["ioctl", "clone", "unshare", "setns", "seccomp", "perf_event_open", "io_uring_setup", "syscall", "mmap",
          "dlopen", "reboot"],
    .07: ["ptrace"],
}



def __init_elf():
    _p, _rn = cached_property, lambda sk: _FLAG_RENAME.get(sk, sk).lower().replace("_", "")
    sec_flags = {k: v for k, v in getattr(lief.ELF.Section.FLAGS, "_member_map_").items()}
    sec_types = {k: v for k, v in getattr(lief.ELF.Section.TYPE, "_member_map_").items()}
    is_ = {f'is_{_rn(k)}': _make_property(k) for k in _FLAGS}
    ELFSection = get_part_class("ELFSection",
        flags="flags",
        flags_str=_p(lambda s: "".join(["", v][getattr(s, f"is_{_rn(k)}")] for k, v in _FLAGS.items())),
        is_code=_p(lambda s: s.flags & 0x4 > 0),
        is_data=_p(lambda s: s.flags & 0x1 > 0 and s.flags & 0x2 > 0 and s.flags & 0x4 == 0),
        raw_data_size=_p(lambda s: len(s.content)),
        virtual_size="size",
        FLAGS=sec_flags,
        TYPES=sec_types,
        **is_,
    )
    seg_flags = {k: v for k, v in getattr(lief.ELF.Segment.FLAGS, "_member_map_").items()}
    seg_types = {k: v for k, v in getattr(lief.ELF.Segment.TYPE, "_member_map_").items()}
    ELFSegment = get_part_class("ELFSegment",
        alignment="alignment",
        offset="file_offset",
        physical_address="physical_address",
        physical_size="physical_size",
        type="type",
        FLAGS=seg_flags,
        TYPES=seg_types,
    )
    
    class ELF(Binary):
        checksum = 0
        DATA = ".data"
        TEXT = ".text"
        
        def __iter__(self):
            for s in self._parsed.sections:
                yield ELFSection(s, self)
        
        def _get_builder(self):
            return lief.ELF.Builder(self._parsed)
        
        @property
        def entrypoint(self):
            return self.virtual_address_to_offset(self._parsed.entrypoint)
        
        @property
        def entrypoint_section(self):
            return ELFSection(self.section_from_offset(self._parsed.entrypoint), self)
        
        @property
        def has_rpath(self):
            return self._parsed.get(lief.ELF.DynamicEntry.TAG.RPATH)
        
        @property
        def has_runpath(self):
            return self._parsed.get(lief.ELF.DynamicEntry.TAG.RUNPATH)
        
        @property
        def is_glibc(self):
            return self.has_interpreter and re.match(r"/lib(?:64)?/ld-linux.*?\.so\.\d+$", self.interpreter) is not None
        
        @property
        def is_musl(self):
            return self.has_interpreter and re.match(r"/lib(?:64)?/ld-musl.*?\.so\.\d+$", self.interpreter) is not None
        
        @property
        def paths(self):
            from re import search
            return [s for s in self.strings if "/" in s and not (s.startswith("//") and s.endswith("//")) and \
                                               search(r"(^#|\s?/\s+|\[./.\]|[^\\]\s+|%.|/--?|=N/A$)", s) is None and \
                                               not ("://" in s and not s.startswith("file://"))]
        
        @property
        def portability(self):
            score = 1.
            # architecture
            if self.header.machine_type not in [lief.ELF.ARCH.X86_64, lief.ELF.ARCH.I386, lief.ELF.ARCH.AARCH64]:
                score -= .2
            # symbols
            penalties = {s: p for p, sl in _SYMBOL_PENALTIES.items() for s in sl}
            score -= min(.2, sum(p for s, p in penalties.items() if s in self.symbols))
            # statically/dynamically linked
            if any(s.name == ".dynamic" for s in self._parsed.sections):
                n_non_std_libs = len([l for l in self.libraries if not l.startswith("libc") and "linux" not in l])
                score -= .1 + min(.1, .02 * n_non_std_libs)
            # ABI
            if self.header.identity_abi_version != lief.ELF.Header.OS_ABI.SYSTEMV:
                score -= .1
            # relocations
            if not self.is_pie:
                score -= .1
            # interpreter
            if not self.is_glibc and not self.is_musl:
                score -= .1
            # RPATH/RUNPATH
            if self.has_rpath or self.has_runpath:
                score -= .05
            # hardcoded paths
            if len(self.paths) > 0:
                score -= .05
            return round(max(0., min(1., score)), 3)
        
        @property
        def segments(self):
            return [ELFSegment(s, self) for s in self._parsed.segments]
        
        @property
        def size_of_header(self):
            return self._parsed.header.program_header_offset + \
                   self._parsed.header.program_header_size * self._parsed.header.numberof_segments
        
        @property
        def symbols(self):
            return list(map(lambda s: s.name, self._parsed.symbols))
    
    ELF.__name__ = "ELF"
    ELF.SECTION_FLAGS = sec_flags
    ELF.SECTION_TYPES = sec_types
    ELF.SEGMENT_FLAGS = seg_flags
    ELF.SEGMENT_TYPES = seg_types
    return ELF
lazy_load_object("ELF", __init_elf)

