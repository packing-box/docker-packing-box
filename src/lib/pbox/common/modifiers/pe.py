import lief

__all__ = ["section_name", "add_section", "SECTION_TYPES", "SECTION_CHARACTERISTICS"]

SECTION_CHARACTERISTICS = lief.PE.SECTION_CHARACTERISTICS.__entries
SECTION_CHARACTERISTICS.update((k,SECTION_CHARACTERISTICS[k][0].value) 
                               for k in SECTION_CHARACTERISTICS)
SECTION_TYPES = lief.PE.SECTION_TYPES.__entries
SECTION_TYPES.update((k,SECTION_TYPES[k][0]) for k in SECTION_TYPES)


def section_name(old_section, new_name):
    """Modifier that renames one of the sections of a binary parsed by lief
    Raises:
        LookupError: section name not found in the binary
    """

    if isinstance(old_section, lief.PE.Section):
        old_name = old_section.name
    else: # if not a Section, assume it is a string
        old_name = old_section
        

    def _section_name(parsed=None, **kw):
        if parsed is None:
            print("A parsed executable must be provided !")
        if not isinstance(parsed, lief.PE.Binary):
            print("Only the lief parser can be used for this function")

        sec = parsed.get_section(old_name)
        if sec is None:
            raise LookupError(f"Section {old_name} not found")
        sec.name = new_name
    return _section_name


def add_section(name,
                section_type=SECTION_TYPES["TEXT"],
                characteristics=SECTION_CHARACTERISTICS["MEM_READ"] +
                SECTION_CHARACTERISTICS["MEM_WRITE"] +
                SECTION_CHARACTERISTICS["MEM_EXECUTE"], 
                data=b""):
    """Modifier that adds a section to a binary parsed by lief
    Wrapper for lief.PE.Binary.add_section
    """

    def _add_section(parsed=None, **kw):
        if parsed is None:
            raise ValueError("A parsed executable must be provided !")
        if not isinstance(parsed, lief.PE.Binary):
            raise TypeError("Only the lief parser can be used for this function")

        # sec = lief.PE.Section(name=name, content=list(data), characteristics=characteristics)
        # for some reason, the above API raises a warning in lief:
        # **[section name] content size is bigger than section's header size**
        # source: https://github.com/lief-project/LIEF/blob/master/src/PE/Builder.cpp
        
        sec = lief.PE.Section(name=name)
        sec.content = list(data)
        sec.characteristics=characteristics
        parsed.add_section(sec, section_type)
    
    return _add_section
