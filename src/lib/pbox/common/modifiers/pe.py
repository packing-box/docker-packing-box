import lief
from lief.PE import SECTION_TYPES
from types import SimpleNamespace

__all__ = ["section_name", "add_section", "SECTION_TYPES"]

d = lief.PE.SECTION_CHARACTERISTICS.__entries
SECTION_CHARACTERISTICS = SimpleNamespace(**{c: d[c][0].value for c in d})

def section_name(old_name, new_name):
    """Modifier that renames one of the sections of a binary parsed by lief
    Raises:
        LookupError: section name not found in the binary
    """

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
                section_type=SECTION_TYPES.TEXT,
                characteristics=SECTION_CHARACTERISTICS.MEM_READ +
                SECTION_CHARACTERISTICS.MEM_WRITE +
                SECTION_CHARACTERISTICS.MEM_EXECUTE, 
                data=b""):
    """Modifier that adds a section to a binary parsed by lief
    Wrapper for lief.PE.Binary.add_section
    """

    def _add_section(parsed=None, **kw):
        if parsed is None:
            raise ValueError("A parsed executable must be provided !")
        if not isinstance(parsed, lief.PE.Binary):
            raise TypeError("Only the lief parser can be used for this function")

        sec = lief.PE.Section(name=name, content=list(data),
                              characteristics=characteristics)
        
        parsed.add_section(sec, section_type)
    
    return _add_section
