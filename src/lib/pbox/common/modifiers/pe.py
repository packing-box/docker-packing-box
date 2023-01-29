import lief
from .parsers import parser_handler


__all__ = ["section_name", "add_section", "append_to_section", "move_entrypoint_to_new_section",
           "move_entrypoint_to_slack_space", "add_API_to_IAT", "add_lib_to_IAT",
           "sections_with_slack_space", "sections_with_slack_space_entry_jump",
           "SECTION_TYPES", "SECTION_CHARACTERISTICS"]

##############################################################################
###############################    Utils   ###################################
##############################################################################

SECTION_CHARACTERISTICS = lief.PE.SECTION_CHARACTERISTICS.__entries
SECTION_CHARACTERISTICS.update((k, SECTION_CHARACTERISTICS[k][0].value)
                               for k in SECTION_CHARACTERISTICS)
SECTION_TYPES = lief.PE.SECTION_TYPES.__entries
SECTION_TYPES.update((k, SECTION_TYPES[k][0]) for k in SECTION_TYPES)


def parse_section_name(section_input):
    """Helper to to allow the user to use sections or section names as input

    Args:
        section_input (lief.PE.Section or str): The section as set by the user

    Raises:
        TypeError: Section must be lief.PE.Section or str

    Returns:
        str: Section name
    """
    if isinstance(section_input, lief.PE.Section):
        return section_input.name
    elif isinstance(section_input, str):
        return section_input
    else:
        raise TypeError(
            f"Section must be lief.PE.Section or str, not {section_input.__class__}")


def sections_with_slack_space(sections, l=1):
    return [s for s in sections if s.size - len(s.content) >= l]


def sections_with_slack_space_entry_jump(sections, pre_data_len=0):
    return sections_with_slack_space(sections, 6 + pre_data_len)


##############################################################################
#############################    Modifiers   #################################
##############################################################################

def section_name(old_section, new_name):
    """Modifier that renames one of the sections of a binary parsed by lief

    Raises:
        LookupError: section name not found in the binary

    Returns:
        function: modifier function
    """

    old_name = parse_section_name(old_section)

    @parser_handler("lief_parser")
    def _section_name(parsed=None, **kw):
        # if parsed is None:
        #    print("A parsed executable must be provided !")
        # if not isinstance(parsed, lief.PE.Binary):
        #    print("Only the lief parser can be used for this function")

        sec = parsed.get_section(old_name)
        if sec is None:
            raise LookupError(f"Section {old_name} not found")
        sec.name = new_name
    return _section_name


def add_section(name,
                section_type=SECTION_TYPES["TEXT"],
                characteristics=SECTION_CHARACTERISTICS["MEM_READ"] +
                SECTION_CHARACTERISTICS["MEM_EXECUTE"],
                data=b""):
    """Modifier that adds a section to a binary parsed by lief
        Wrapper for lief.PE.Binary.add_section

    Args:
        name (str): Name of the new section
        section_type (lief.PE.SECTION_TYPE, optional): Type of the new section. Defaults to SECTION_TYPES["TEXT"].
        characteristics (lief.PE.SECTION_CHARACTERISTICS, optional): Characteristics of the new section. Defaults to SECTION_CHARACTERISTICS["MEM_READ"]+SECTION_CHARACTERISTICS["MEM_EXECUTE"].
        data (bytes, optional): Content of the new section. Defaults to b"".

    Returns:
        function: modifier function
    """

    @parser_handler("lief_parser")
    def _add_section(parsed=None, **kw):
        # if parsed is None:
        #    raise ValueError("A parsed executable must be provided !")
        # if not isinstance(parsed, lief.PE.Binary):
        #    raise TypeError("Only the lief parser can be used for this function")

        # sec = lief.PE.Section(name=name, content=list(data), characteristics=characteristics)
        # for some reason, the above API raises a warning in lief:
        # **[section name] content size is bigger than section's header size**
        # source: https://github.com/lief-project/LIEF/blob/master/src/PE/Builder.cpp

        sec = lief.PE.Section(name=name)
        sec.content = list(data)
        sec.characteristics = characteristics
        parsed.add_section(sec, section_type)

    return _add_section


def append_to_section(section_input, data_source):
    """Modifier that appends bytes at the end of a section, in the slack space
        before the next section

    Args:
        section_input (lief.PE.Section or str): The section to append bytes to.
        data_source (Callable object or bytes): The data to append. If it is a
            callable object, it is expected to take as single argument the size
            the available space between the sections and to return the data to 
            append, as bytes.

    Returns:
        function: modifier function
    """
    section_name = parse_section_name(section_input)

    @parser_handler("lief_parser")
    def _append_to_section(parsed=None, **kw):
        # if parsed is None:
        #    print("A parsed executable must be provided !")
        # if not isinstance(parsed, lief.PE.Binary):
        #    print("Only the lief parser can be used for this function")

        section = parsed.get_section(section_name)
        available_size = section.size - len(section.content)

        if callable(data_source):
            data = list(data_source(available_size))
        else:
            data = list(data_source)

        if len(data) > available_size:
            print(f"""Warning: Data length is more than the available space at the end of the section.
                          The slack space is limited to {available_size} bytes, {len(data)} bytes of data were provided.
                          Data will be truncated to fit in the slack space.""")
            data = data[:available_size]
        section.content = list(section.content) + data
    return _append_to_section


def move_entrypoint_to_new_section(name,
                                   section_type=SECTION_TYPES["TEXT"],
                                   characteristics=SECTION_CHARACTERISTICS["MEM_READ"] +
                                   SECTION_CHARACTERISTICS["MEM_EXECUTE"],
                                   pre_data=b"",
                                   post_data=b""):
    """Modifier that sets the entrypoint to a new section added to the binary, 
        that contains code to jump back to the original entrypoint.
        The new section contains *pre_data*, then the code to jump back to the original
        entrypoint, and finally *post_data*.

    Args:
        name (str): Name of the new section
        section_type (lief.PE.SECTION_TYPE, optional): Type of the new section. Defaults to SECTION_TYPES["TEXT"].
        characteristics (lief.PE.SECTION_CHARACTERISTICS, optional): Characteristics of the new section. Defaults to SECTION_CHARACTERISTICS["MEM_READ"]+SECTION_CHARACTERISTICS["MEM_EXECUTE"].
        pre_data (bytes, optional): Data to prepend to the new section. Defaults to b"".
        post_data (bytes, optional): Data to append to the new section. Defaults to b"".
    """

    @parser_handler("lief_parser")
    def _move_entrypoint_to_new_section(parsed=None, **kw):

        old_entry = parsed.entrypoint + 0x10000
        # push current_entrypoint
        # ret
        entrypoint_data = [0x68] + \
            list(old_entry.to_bytes(4, 'little')) + [0xc3]

        # Other possibility:
        # mov eax current_entrypoint
        # jmp eax
        # entrypoint_data = [0xb8] + list(old_entry.to_bytes(4, 'little')) + [0xff, 0xe0]

        full_data = list(pre_data) + entrypoint_data + list(post_data)

        add_section(name,
                    section_type=section_type,
                    characteristics=characteristics,
                    data=full_data)(parsed=parsed)

        new_entry = parsed.get_section(name).virtual_address + len(pre_data)
        parsed.optional_header.addressof_entrypoint = new_entry

    return _move_entrypoint_to_new_section


def move_entrypoint_to_slack_space(section_input,
                                   pre_data=b"",
                                   post_data_source=b""):
    """Modifier that sets the entrypoint to a new section added to the binary, 
        that contains code to jump back to the original entrypoint.

    Args:
        section_input (lief.PE.Section or str): The section to append bytes to.
        pre_data (bytes): The data to prepend.
        post_data_source (Callable object or bytes): The data to append. If it is a
            callable object, it is expected to take as single argument the size
            the available space and to return the data to append, as bytes.
    """
    section_name = parse_section_name(section_input)

    @parser_handler("lief_parser")
    def _move_entrypoint_to_slack_space(parsed=None, **kw):

        old_entry = parsed.entrypoint + 0x10000
        # push current_entrypoint
        # ret
        entrypoint_data = [0x68] + \
            list(old_entry.to_bytes(4, 'little')) + [0xc3]

        # Other possibility:
        # mov eax current_entrypoint
        # jmp eax
        # entrypoint_data = [0xb8] + list(old_entry.to_bytes(4, 'little')) + [0xff, 0xe0]

        d = list(pre_data) + entrypoint_data

        if callable(post_data_source):
            def full_data(l): return d + list(post_data_source(l - len(d)))
        else:
            full_data = d + list(post_data_source)

        section = parsed.get_section(section_name)
        new_entry = section.virtual_address + \
            len(section.content) + len(pre_data)

        append_to_section(section_name,
                          data_source=full_data)(parsed=parsed)

        parsed.optional_header.addressof_entrypoint = new_entry

    return _move_entrypoint_to_slack_space


def add_API_to_IAT(API, lib_name):
    """Modifier that adds a function to the IAT. 
        If no function from this library is imported in the binary yet, the
        library is added to the binary.

    Args:
        API (str): Function name to add to the IAT
        lib_name (str): Name of the library that contains the function 
    """

    @parser_handler("lief_parser")
    def _add_API_to_IAT(parsed, build_instructions, **kw):
        build_instructions.update(imports=True)
        for library in parsed.imports:
            if library.name.lower() == lib_name.lower():
                library.add_entry(API)
                return
        library = add_lib_to_IAT(lib_name)(parsed=parsed)
        library.add_entry(API)

    return _add_API_to_IAT


def add_lib_to_IAT(lib_name):
    """Modifier that adds a library to the IAT

    Args:
        lib_name (str): Name of the library to add to the IAT
    """

    @parser_handler("lief_parser")
    def _add_lib_to_IAT(parsed, build_instructions, **kw):
        build_instructions.update(imports=True)
        return parsed.add_library(lib_name)

    return _add_lib_to_IAT
