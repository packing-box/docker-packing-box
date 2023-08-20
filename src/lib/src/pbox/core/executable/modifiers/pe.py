# -*- coding: UTF-8 -*-
import lief
from ..parsers import *


__all__ = [
   "get_pe_data", "SECTION_CHARACTERISTICS", "SECTION_TYPES",
    # utils
   "content_size", "has_characteristic", "raw_data_size", "section_name", "sections_with_slack_space",
   "sections_with_slack_space_entry_jump", "virtual_size",
    # modifiers
   "add_API_to_IAT", "add_lib_to_IAT", "add_section", "append_to_section", "append_to_section_force",
   "move_entrypoint_to_new_section", "move_entrypoint_to_slack_space", "nop", "rename_all_sections", "rename_section",
   "set_checksum",
]


# --------------------------------------------------- Utils ------------------------------------------------------------
SECTION_CHARACTERISTICS = {k: lief.PE.SECTION_CHARACTERISTICS.__entries[k][0] for k in \
                           lief.PE.SECTION_CHARACTERISTICS.__entries}
SECTION_TYPES = {k: lief.PE.SECTION_TYPES.__entries[k][0] for k in lief.PE.SECTION_TYPES.__entries}

content_size                         = lambda s: len(s.content)
has_characteristic                   = lambda s, c: s.has_characteristic(c)
raw_data_size                        = lambda s: s.size
section_name                         = lambda s: s.name
sections_with_slack_space            = lambda s, l=1: [_s for _s in s if _s.size - len(_s.content) >= l]
sections_with_slack_space_entry_jump = lambda s, dl=0: sections_with_slack_space(s, 6 + dl)
virtual_size                         = lambda s: s.virtual_size


def get_pe_data():
    """ Derive other PE-specific data from this of ~/.opt/data/pe. """
    from ....helpers.data import get_data
    d = get_data("PE")['COMMON_DLL_IMPORTS']
    d = {'COMMON_API_IMPORTS': [(lib, api) for lib in d for api in d[lib]]}
    return d


# ------------------------------------------------- Modifiers ----------------------------------------------------------
# Conventions:
#  - modifiers are decorators to be called either on a parsed binary or given a pair (parser, executable)
#  - modifiers are named starting with a verb and describe the action performed

def add_API_to_IAT(*args):
    """ Add a function to the IAT. If no function from this library is imported in the binary yet, the library is added
         to the binary.
    
    :param args: either (library, api) or ((library, api), )
    """
    if len(args) == 1:
        lib_name, api = args[0]
    elif len(args) == 2:
        lib_name, api = args
    else:
        raise ValueError("A lib name and an API name must be provided")
    @parser_handler("lief_parser")
    def _add_API_to_IAT(parsed=None, executable=None, **kw):
        iat = parsed.data_directory(lief.PE.DATA_DIRECTORY.IMPORT_TABLE)
        patch_imports = False
        # Some packers create the IAT at runtime. It is sometimes in a empty section, which has offset 0. In this case,
        #  the header is overwritten by the patching operation. So, in this case, we don't patch at all.
        if iat.has_section:
            if iat.section.offset == 0:
                patch_imports = False
        for library in parsed.imports:
            if library.name.lower() == lib_name.lower():
                library.add_entry(api)
                return {"imports": True, "patch_imports": patch_imports}
        add_lib_to_IAT(lib_name)(parsed=parsed)
        parsed.get_import(lib_name).add_entry(api)
        return {"imports": True, "patch_imports": patch_imports}
    return _add_API_to_IAT


def add_lib_to_IAT(library):
    """ Add a library to the IAT. """
    @parser_handler("lief_parser")
    def _add_lib_to_IAT(parsed, **kw):
        parsed.add_library(library)
        return {"imports": True}
    return _add_lib_to_IAT


def add_section(name, section_type=SECTION_TYPES["UNKNOWN"], characteristics=SECTION_CHARACTERISTICS["MEM_READ"] |
                SECTION_CHARACTERISTICS["MEM_EXECUTE"], data=b""):
    """ Add a section (uses lief.PE.Binary.add_section).
    
    :param name:            name of the new section
    :param section_type:    type of the new section (lief.PE.SECTION_TYPE)
    :param characteristics: characteristics of the new section
    :param data:            content of the new section
    :return:                modifier function
    """
    if len(name) > 8:
        raise ValueError("Section name can't be longer than 8 characters")
    @parser_handler("lief_parser")
    def _add_section(parsed=None, **kw):
        # sec = lief.PE.Section(name=name, content=list(data), characteristics=characteristics)
        # for some reason, the above API raises a warning in LIEF:
        # **[section name] content size is bigger than section's header size**
        # source: https://github.com/lief-project/LIEF/blob/master/src/PE/Builder.cpp
        s = lief.PE.Section(name=name)
        s.content = list(data)
        s.characteristics = characteristics
        parsed.add_section(s, section_type)
    return _add_section


def append_to_section(section_input, data):
    """ Append bytes (either raw data or a function run with the target section's available size in the slack space as
         its single parameter) at the end of a section, in the slack space before the next section.
    """
    @parser_handler("lief_parser")
    def _wrapper(parsed=None, **kw):
        section = parse_section(section_input, parsed)
        available_size = section.size - len(section.content)
        d = list(data(available_size)) if callable(data) else list(data)
        if len(d) > available_size:
            print("""Warning: Data length is greater than the available space at the end of the section.
                          The slack space is limited to %d bytes, %d bytes of data were provided.
                          Data will be truncated to fit in the slack space.""" % (available_size, len(d)))
            d = d[:available_size]
        section.content = list(section.content) + d
    return _wrapper


def append_to_section_force(section_input, data):
    """ Append bytes at the end of a section (uses lief.PE.Section) even it is larger than the slack space before the
         next section. Not that it often increases the size of the section on disk.
    """
    @parser_handler("lief_parser")
    def _wrapper(parsed=None, **kw):
        if len(data) == 0:
            return
        section = parse_section(section_input, parsed)
        sec_name = section.name
        sections_dict = []
        sections = sorted(list(parsed.sections), key=lambda x: x.virtual_address)
        for s in sections:
            d = {'name': s.name, 'virtual_address': s.virtual_address, 'virtual_size': s.virtual_size,
                 'char': s.characteristics}
            d['content'] = list(s.content) + (list(data) if s == section else [])
        for s in sections:
            parsed.remove(s)
        for st in sections_dict:
            c = st['content']
            fa = parsed.optional_header.file_alignment
            in_sec = lief.PE.Section(content=c + [0] * (-len(c) % fa), name=st['name'], characteristics=st['char'])
            new_sec = parsed.add_section(in_sec)
            new_sec.content = c
            new_sec.virtual_size = st['virtual_size']
            new_sec.virtual_address = st['virtual_address']
    return _wrapper


def move_entrypoint_to_new_section(name, section_type=SECTION_TYPES["UNKNOWN"],
                                   characteristics=SECTION_CHARACTERISTICS["MEM_READ"] |
                                   SECTION_CHARACTERISTICS["MEM_EXECUTE"],
                                   pre_data=b"", post_data=b""):
    """ Set the entrypoint (EP) to a new section added to the binary that contains code to jump back to the original EP.
        The new section contains *pre_data*, then the code to jump back to the original EP, and finally *post_data*.
    """
    @parser_handler("lief_parser")
    def _wrapper(parsed=None, executable=None, **kw):
        address_bitsize = [64, 32]["32" in executable.format]
        old_entry = parsed.entrypoint + 0x10000
        #  push current_entrypoint
        #  ret
        entrypoint_data = [0x68] + list(old_entry.to_bytes(address_bitsize//8, 'little')) + [0xc3]
        # other possibility:
        #  mov eax current_entrypoint
        #  jmp eax
        #  entrypoint_data = [0xb8] + list(old_entry.to_bytes(4, 'little')) + [0xff, 0xe0]
        full_data = list(pre_data) + entrypoint_data + list(post_data)
        add_section(name, section_type=section_type, characteristics=characteristics, data=full_data)(parsed=parsed)
        new_entry = parsed.get_section(name).virtual_address + len(pre_data)
        parsed.optional_header.addressof_entrypoint = new_entry
    return _wrapper


def move_entrypoint_to_slack_space(section_input, pre_data=b"", post_data_source=b""):
    """ Set the entrypoint (EP) to a new section added to the binary that contains code to jump back to the original EP.
    """
    @parser_handler("lief_parser")
    def _wrapper(parsed=None, executable=None, **kw):
        if parsed.optional_header.section_alignment % parsed.optional_header.file_alignment != 0:
            raise ValueError("SectionAlignment is not a multiple of FileAlignment. File integrity cannot be assured.")
        address_bitsize = [64, 32]["32" in executable.format]
        old_entry = parsed.entrypoint + 0x10000
        #  push current_entrypoint
        #  ret
        entrypoint_data = [0x68] + list(old_entry.to_bytes(address_bitsize//8, 'little')) + [0xc3]
        # other possibility:
        #  mov eax current_entrypoint
        #  jmp eax
        #  entrypoint_data = [0xb8] + list(old_entry.to_bytes(4, 'little')) + [0xff, 0xe0]
        d = list(pre_data) + entrypoint_data
        if callable(post_data_source):
            full_data = lambda l: d + list(post_data_source(l - len(d)))
            add_size = section.size - len(section.content)
        else:
            full_data = d + list(post_data_source)
            add_size = len(full_data)
        section = parse_section(section_input, parsed)
        new_entry = section.virtual_address + len(section.content) + len(pre_data)
        append_to_section(section, data_source=full_data)(parsed=parsed)
        section.virtual_size = section.virtual_size + add_size
        parsed.optional_header.addressof_entrypoint = new_entry
    return _wrapper


def nop():
    """ Do nothing. """
    def _nop(**kw):
        pass
    return _nop


def rename_all_sections(old_sections, new_sections):
    """ Rename a given list of sections. """
    modifiers = [rename_section(old, new) for old, new in zip(old_sections, new_sections)]
    def _wrapper(parsed, **kw):
        for m in modifiers:
            try:
                parser = m(parsed, **kw)
            except LookupError:
                parser = None
        return parser
    return _wrapper


def rename_section(old_section, new_name, raise_error=False):
    """ Rename a given section. """
    if len(new_name) > 8:
        raise ValueError("Section name can't be longer than 8 characters")
    @parser_handler("lief_parser")
    def _wrapper(parsed=None, **kw):
        sec = parse_section(old_section, parsed)
        old_name = sec.name
        if sec is None:
            if raise_error:
                raise LookupError("Section %s not found" % old_name)
            else:
                return
        sec.name = new_name
    return _wrapper


def set_checksum(value):
    """ Set the checksum. """
    @parser_handler("lief_parser")
    def _set_checksum(parsed, **kw):
        parsed.optional_header.checksum = value
    return _set_checksum

