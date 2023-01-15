import lief
from .parsers import *


__all__ = ["section_name", "add_section",
           "SECTION_TYPES", "SECTION_CHARACTERISTICS"]

SECTION_CHARACTERISTICS = lief.PE.SECTION_CHARACTERISTICS.__entries
SECTION_CHARACTERISTICS.update((k, SECTION_CHARACTERISTICS[k][0].value)
                               for k in SECTION_CHARACTERISTICS)
SECTION_TYPES = lief.PE.SECTION_TYPES.__entries
SECTION_TYPES.update((k, SECTION_TYPES[k][0]) for k in SECTION_TYPES)


def section_name(old_section, new_name):
    """Modifier that renames one of the sections of a binary parsed by lief
    
    Raises:
        LookupError: section name not found in the binary
    
    Returns:
        function: modifier function
    """

    if isinstance(old_section, lief.PE.Section):
        old_name = old_section.name
    else:  # if not a Section, assume it is a string
        old_name = old_section

    @lief_parser
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


    @lief_parser
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

    @lief_parser
    def _append_to_section(parsed=None, **kw):
        # if parsed is None:
        #    print("A parsed executable must be provided !")
        # if not isinstance(parsed, lief.PE.Binary):
        #    print("Only the lief parser can be used for this function")

        if isinstance(section_input, lief.PE.Section):
            section = section_input
        else:
            section = parsed.get_section(section_input)

        available_size = section.size - len(section.content)

        if callable(data_source):
            data = data_source(available_size)
        else:
            data = list(data_source)

        if len(data) > available_size:
            print(f"""Warning: Data length is more than the available space at the end of the section.
                          The slack space is limited to {available_size} bytes, {len(data)} bytes of data were provided.
                          Data will be truncated to fit in the slack space.""")
            data = data[:available_size]
        section.content += data
    return _append_to_section
