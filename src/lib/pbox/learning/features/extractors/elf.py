# -*- coding: UTF-8 -*-
from elftools.elf.elffile import ELFFile


__all__ = ["elfeats", "ELFEATS", "STD_SECTIONS"]


ELFEATS = {
    'magic': "Sum of magic bytes",
    # 7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00 
    # 7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00 
    # 7f 45 4c 46 02 01 01 03 00 00 00 00 00 00 00 00 
    #  NB: the sum of these numbers give distinct results, so no loss of information
    'data': "Data",
    'type': "Type",
    'machine': "Machine",
    'version': "Version",
    'entry_address': "Entry point address",
    'abi_version': "OS/ABI",
    'flags': "Flags",
    'size_of_elf_header': "Size of ELF header",
    'start_prg_headers': "Start of program headers",
    'size_prg_headers': "Size of program headers",
    'number_prg_headers': "Number of program headers",
    'start_sec_headers': "Start of section headers",
    'size_sec_headers': "Size of section headers",
    'number_sec_headers': "Number of section headers",
    'sec_header_str_table_idx': "Section header string table index",
}
# This list is computed taking binaries into account from:
#  - https://www.cs.cmu.edu/afs/cs/academic/class/15213-f00/docs/elf.pdf (specification)
#  - /bin (Ubuntu)
STD_SECTIONS = """
.bss
.comment
.comment.SUSE.OPTs
.conflict
.ctors
.data
.data.rel.ro
.data.rel.ro.local
.data1
.debug
.debug_abbrev
.debug_aranges
.debug_frame
.debug_gdb_scripts
.debug_info
.debug_line
.debug_loc
.debug_pubnames
.debug_ranges
.debug_str
.dtors
.dynamic
.dynstr
.dynsym
.eh_frame
.eh_frame_hdr
.fini
.fini_array
.gcc_except_table
.gnu.hash
.gnu.version
.gnu.version_r
.gnu_debugaltlink
.gnu_debuglink
.go.buildinfo
.gopclntab
.gosymtab
.got
.got.plt
.gptab
.gresource.applet
.gresource.baobab
.gresource.bolt_dbus
.gresource.cc_applications
.gresource.cc_background
.gresource.cc_bluetooth
.gresource.cc_color
.gresource.cc_common
.gresource.cc_connectivity
.gresource.cc_datetime
.gresource.cc_default_apps
.gresource.cc_diagnostics
.gresource.cc_display
.gresource.cc_info_overview
.gresource.cc_keyboard
.gresource.cc_location
.gresource.cc_lock
.gresource.cc_mouse
.gresource.cc_network
.gresource.cc_notifications
.gresource.cc_online_accounts
.gresource.cc_power
.gresource.cc_printers
.gresource.cc_region
.gresource.cc_removable_media
.gresource.cc_search
.gresource.cc_sharing
.gresource.cc_sound
.gresource.cc_thunderbolt
.gresource.cc_ubuntu
.gresource.cc_universal_access
.gresource.cc_usage
.gresource.cc_user_accounts
.gresource.cc_wacom
.gresource.ce
.gresource.dconf_editor
.gresource.ev
.gresource.ev_previewer
.gresource.font_view
.gresource.fr
.gresource.fu
.gresource.ghex
.gresource.gimagereader
.gresource.gl
.gresource.gnome_calculator
.gresource.gnome_control_center
.gresource.gnome_disks
.gresource.gnome_extensions_tool
.gresource.gpm
.gresource.gs
.gresource.gth
.gresource.hdy
.gresource.logview
.gresource.nautilus
.gresource.net_connection_editor
.gresource.peek
.gresource.resources
.gresource.screenshot
.gresource.seahorse
.gresource.wireless_security
.hash
.init
.init_array
.interp
.itablink
.jcr
.liblist
.line
.lit4
.lit8
.module_license
.noptrbss
.noptrdata
.note
.note.ABI-tag
.note.SuSE
.note.crashpad.info
.note.gnu.build-id
.note.gnu.gold-version
.note.gnu.property
.note.go.buildid
.note.stapsdt
.plt
.plt.got
.plt.sec
.probes
.qtversion
.reginfo
.rel.bss
.rel.comment
.rel.conflict
.rel.data
.rel.data1
.rel.debug
.rel.dyn
.rel.dynamic
.rel.dynstr
.rel.fini
.rel.fini_array
.rel.gnu_debuglink
.rel.got
.rel.gptab
.rel.hash
.rel.init
.rel.init_array
.rel.interp
.rel.liblist
.rel.line
.rel.lit4
.rel.lit8
.rel.note
.rel.plt
.rel.reginfo
.rel.rodata
.rel.rodata1
.rel.sbss
.rel.sdata
.rel.shstrtab
.rel.strtab
.rel.symtab
.rel.tdesc
.rel.text
.rela
.rela.bss
.rela.comment
.rela.conflict
.rela.data
.rela.data1
.rela.debug
.rela.dyn
.rela.dynamic
.rela.dynstr
.rela.fini
.rela.fini_array
.rela.gnu_debuglink
.rela.got
.rela.gptab
.rela.hash
.rela.init
.rela.init_array
.rela.interp
.rela.liblist
.rela.line
.rela.lit4
.rela.lit8
.rela.note
.rela.plt
.rela.reginfo
.rela.rodata
.rela.rodata1
.rela.sbss
.rela.sdata
.rela.shstrtab
.rela.strtab
.rela.symtab
.rela.tdesc
.rela.text
.rodata
.rodata1
.sbss
.sdata
.shstrtab
.stapsdt.base
.strtab
.symtab
.tbss
.tdata
.tdesc
.text
.tm_clone_table
.typelink
.zdebug_abbrev
.zdebug_aranges
.zdebug_frame
.zdebug_info
.zdebug_line
.zdebug_loc
.zdebug_ranges
.zdebug_str
SYSTEMD_BUS_ERROR_MAP
SYSTEMD_STATIC_DESTRUCT
__debug
__libc_IO_vtables
__libc_atexit
__libc_freeres_fn
__libc_freeres_ptrs
__libc_subfreeres
google_malloc
malloc_hook
""".strip().split("\n")


def elfeats(executable):
    """ This uses features computed in ELF-Miner to extract +300 features from ELF files. """
    with executable.open('rb') as fh:
        elf = ELFFile(fh)
        headers = elf.header
    return {
        'magic': sum(elf.e_ident_raw),
        'data': headers['e_ident']['EI_DATA'],
        'type': headers['e_type'],
        'machine': headers['e_machine'],
        'version': headers['e_ident']['EI_VERSION'],
        'entry_address': headers['e_entry'],
        'os_abi': headers['e_ident']['EI_OSABI'],
        'abi_version': headers['e_ident']['EI_ABIVERSION'],
        'flags': headers['e_flags'],
        'size_of_elf_header': headers['e_ehsize'],
        'start_prg_headers': headers['e_phoff'],
        'size_prg_headers': headers['e_phentsize'],
        'number_prg_headers': headers['e_phnum'],
        'start_sec_headers': headers['e_shoff'],
        'size_sec_headers': headers['e_shentsize'],
        'number_sec_headers': headers['e_shnum'],
        'sec_header_str_table_idx': headers['e_shstrndx'],
    }

