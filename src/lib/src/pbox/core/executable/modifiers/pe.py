# -*- coding: UTF-8 -*-
from ..parsers import *


__all__ = [
    # utils
   "get_pe_data", "valid_names",
    # modifiers
   "add_API_to_IAT", "add_lib_to_IAT", "add_section", "append_to_section", "move_entrypoint_to_new_section",
   "move_entrypoint_to_slack_space", "rename_all_sections", "rename_section", "set_checksum",
]


# --------------------------------------------------- Utils ------------------------------------------------------------
def get_pe_data():
    """ Derive other PE-specific data from this of ~/.packing-box/data/pe. """
    from ....helpers.data import get_data
    d = get_data("PE")['COMMON_DLL_IMPORTS']
    d = {'COMMON_API_IMPORTS': [(lib, api) for lib in d for api in d[lib]]}
    for k in ["COMMON_PACKER_SECTION_NAMES", "STANDARD_SECTION_NAMES"]:
        d[k] = valid_names(d[k])
    return d


valid_names = lambda nl: list(filter(lambda n: len(n) <= 8, map(lambda x: x if isinstance(x, str) else \
                                                                getattr(x, "real_name", getattr(x, "name", "")), nl)))


# ------------------------------------------------- Modifiers ----------------------------------------------------------
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
        raise ValueError("Library and API names shall be provided")
    @supported_parsers("lief")
    def _add_API_to_IAT(parsed, logger):
        logger.debug(" > selected API import: %s - %s" % (lib_name, api))
        # Some packers create the IAT at runtime. It is sometimes in an empty section, which has offset 0. In this case,
        #  the header is overwritten by the patching operation. So, in this case, we don't patch at all.
        patch_imports = not parsed.iat.has_section or parsed.iat.section.offset != 0
        for library in parsed.imports:
            if library.name.lower() == lib_name.lower():
                logger.debug("adding API import...")
                library.add_entry(api)
                parsed._build_config.toggle(imports=True, patch_imports=patch_imports)
                return
        add_lib_to_IAT(lib_name)(parsed, logger)
        parsed.get_import(lib_name).add_entry(api)
        parsed._build_config.toggle(imports=True, patch_imports=patch_imports)
    return _add_API_to_IAT


def add_lib_to_IAT(library):
    """ Add a library to the IAT. """
    @supported_parsers("lief")
    def _add_lib_to_IAT(parsed, logger):
        logger.debug("adding library...")
        parsed.add_library(library)
        parsed._build_config.toggle(imports=True)
    return _add_lib_to_IAT


def add_section(name, section_type=None, characteristics=None, data=b""):
    """ Add a section (uses lief.PE.Binary.add_section).
    
    :param name:            name of the new section
    :param section_type:    type of the new section (lief.PE.SECTION_TYPE)
    :param characteristics: characteristics of the new section
    :param data:            content of the new section
    :return:                modifier function
    """
    if len(name) > 8:
        raise ValueError("Section name can't be longer than 8 characters")
    @supported_parsers("lief")
    def _add_section(parsed, logger):
        # sec = lief.PE.Section(name=name, content=list(data), characteristics=characteristics)
        # for some reason, the above API raises a warning in LIEF:
        #  **[section name] content size is bigger than section's header size**
        # source: https://github.com/lief-project/LIEF/blob/master/src/PE/Builder.cpp
        #from ..parsers import get_parser
        #s = get_parser("lief").PE.Section(name=name)
        from ..parsers.lief.__common__ import lief
        s = lief.PE.Section(name=name)
        s.content = list(data)
        s.characteristics = characteristics or parsed.SECTION_CHARACTERISTICS['MEM_READ'] | \
                                               parsed.SECTION_CHARACTERISTICS['MEM_EXECUTE']
        parsed.add_section(s, section_type or parsed.SECTION_TYPES['UNKNOWN'])
        parsed._build_config.toggle(overlay=True)
    return _add_section


def append_to_section(section, data):
    """ Append bytes (either raw data or a function run with the target section's available size in the slack space as
         its single parameter) at the end of a section, in the slack space before the next section.
    """
    @supported_parsers("lief")
    def _append_to_section(parsed, logger):
        s = parsed.section(name, True)
        available_size = s.size - len(s.content)
        d = list(data(available_size)) if callable(data) else list(data)
        if len(d) > available_size:
            logger.warning("Warning: Data length is greater than the available space at the end of the section. The" \
                           " slack space is limited to %d bytes, %d bytes of data were provided. Data will be " \
                           "truncated to fit in the slack space." % (available_size, len(d)))
            d = d[:available_size]
        s.content = list(s.content) + d
    return _append_to_section


def move_entrypoint_to_new_section(name, section_type=None, characteristics=None, pre_data=b"", post_data=b""):
    """ Set the entrypoint (EP) to a new section added to the binary that contains code to jump back to the original EP.
        The new section contains *pre_data*, then the code to jump back to the original EP, and finally *post_data*.
    """
    @supported_parsers("lief")
    def _move_entrypoint_to_new_section(parsed, logger):
        address_bitsize = [64, 32]["32" in parsed.path.format]
        old_entry = parsed.entrypoint + 0x10000
        #  push current_entrypoint
        #  ret
        entrypoint_data = [0x68] + list(old_entry.to_bytes(address_bitsize // 8, "little")) + [0xc3]
        # other possibility:
        #  mov eax current_entrypoint
        #  jmp eax
        #  entrypoint_data = [0xb8] + list(old_entry.to_bytes(4, 'little')) + [0xff, 0xe0]
        full_data = list(pre_data) + entrypoint_data + list(post_data)
        add_section(name, section_type, characteristics, full_data)(parsed, logger)
        parsed.optional_header.addressof_entrypoint = parsed.get_section(name).virtual_address + len(pre_data)
    return _move_entrypoint_to_new_section


def move_entrypoint_to_slack_space(section_input, pre_data=b"", post_data_source=b""):
    """ Set the entrypoint (EP) to a new section added to the binary that contains code to jump back to the original EP.
    """
    @supported_parsers("lief")
    def _move_entrypoint_to_slack_space(parsed, logger):
        if parsed.optional_header.section_alignment % parsed.optional_header.file_alignment != 0:
            raise ValueError("SectionAlignment is not a multiple of FileAlignment (file integrity cannot be assured)")
        address_bitsize = [64, 32]["32" in parsed.path.format]
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
        s = parsed.section(name, True)
        new_entry = section.virtual_address + len(section.content) + len(pre_data)
        append_to_section(section, full_data)(parsed, logger)
        s.virtual_size += add_size
        parsed.optional_header.addressof_entrypoint = new_entry
    return _move_entrypoint_to_slack_space


def rename_all_sections(old_sections, new_sections):
    """ Rename a given list of sections. """
    modifiers = [rename_section(old, new) for old, new in zip(old_sections, new_sections)]
    def _rename_all_sections(parsed, logger):
        for m in modifiers:
            try:
                parser = m(parsed, logger)
            except LookupError:
                parser = None
        return parser
    return _rename_all_sections


def rename_section(old_section, new_name):
    """ Rename a given section. """
    if old_section is None:
        raise ValueError("Old section shall not be None")
    if new_name is None:
        raise ValueError("New section name shall not be None")
    if len(new_name) > 8:
        raise ValueError("Section name can't be longer than 8 characters (%s)" % new_name)
    def _rename_section(parsed, logger):
        sec = parsed.section(old_section, original=True) if isinstance(old_section, str) else old_section
        logger.debug("rename: %s -> %s" % (sec.name or "<empty>", new_name))
        sec.name = new_name
    return _rename_section


def set_checksum(value):
    """ Set the checksum. """
    @supported_parsers("lief")
    def _set_checksum(parsed, logger):
        parsed.optional_header.checksum = value
    return _set_checksum

