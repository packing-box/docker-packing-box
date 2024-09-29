# -*- coding: UTF-8 -*-
from tinyscript import ensure_str

from ..parsers import *


__all__ = [
    # utils
   "get_pe_data", "valid_names",
    # modifiers
   "add_API_to_IAT", "add_lib_to_IAT", "add_section", "append_to_section", "move_entrypoint_to_new_section",
   "move_entrypoint_to_slack_space", "set_checksum",
]

_DEAD_CODE_BOUNDS = [64, 256]
_DEAD_CODE_MAX_REPEAT = 16


# --------------------------------------------------- Utils ------------------------------------------------------------
_listify = lambda l: [[int(x, 16) for x in s.split(', ')] for s in l]
def get_pe_data():
    """ Derive other PE-specific data from this of ~/.packing-box/data/pe. """
    from ....helpers.data import get_data
    d = {k: v for k, v in get_data("PE").items() if k in ["COMMON_API_IMPORTS", "COMMON_DLL_IMPORTS",
                                                          "COMMON_PACKER_SECTION_NAMES", "STANDARD_SECTION_NAMES", 
                                                          "DEAD_CODE"]}
    d['COMMON_API_IMPORTS'] = [(lib, api) for lib, lst in d.pop('COMMON_DLL_IMPORTS').items() for api in lst]
    for k in ["COMMON_PACKER_SECTION_NAMES", "STANDARD_SECTION_NAMES"]:
        d[k] = valid_names(d[k])
    
    # add DEADCODE
    d['DEAD_CODE'] = _listify(d["DEAD_CODE"]) # will contain a list of lists of bytes [[0x90], [0x33, 0xC0]]

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
        logger.debug(f">> selected API import: {lib_name} - {api}")
        # Some packers create the IAT at runtime. It is sometimes in an empty section, which has offset 0. In this case,
        #  the header is overwritten by the patching operation. So, in this case, we don't patch at all.
        patch_imports = not parsed.iat.has_section or parsed.iat.section.offset != 0
        for library in parsed.imports:
            if library.name.lower() == lib_name.lower():
                logger.debug(">> adding API import...")
                library.add_entry(api)
                parsed._build_config.update(imports=True, patch_imports=patch_imports)
                return
        add_lib_to_IAT(lib_name)(parsed, logger)
        parsed.get_import(lib_name).add_entry(api)
        parsed._build_config.update(imports=True, patch_imports=patch_imports)
    return _add_API_to_IAT


def add_lib_to_IAT(library):
    """ Add a library to the IAT. """
    @supported_parsers("lief")
    def _add_lib_to_IAT(parsed, logger):
        logger.debug(">> adding library...")
        parsed.add_library(library)
        parsed._build_config['imports'] = True
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
        from ..parsers.lief.__common__ import lief
        s = lief.PE.Section(name=name)
        s.content = list(data)
        s.characteristics = characteristics or parsed.SECTION_CHARACTERISTICS['MEM_READ'] | \
                                               parsed.SECTION_CHARACTERISTICS['MEM_EXECUTE']
        parsed.add_section(s, section_type or parsed.SECTION_TYPES['UNKNOWN'])
        parsed._build_config['overlay'] = True
    return _add_section


def append_to_section(name, data):
    """ Append bytes (either raw data or a function run with the target section's available size in the slack space as
         its single parameter) at the end of a section, in the slack space before the next section.
    """
    @supported_parsers("lief")
    def _append_to_section(parsed, logger):
        section = parsed.section(name, True)
        l, fa = len(section.content), parsed.optional_header.file_alignment
        available_size = max(0, section.virtual_size - l)
        d = list(data(available_size)) if callable(data) else list(data)
        logger.debug(f"section: {section.name} - content: {l} - raw size: {section.size} - virtual size: "
                     f"{section.virtual_size} - file alignment: {fa}")
        if available_size == 0:
            logger.debug(f">> {section.name}: no available space")
        elif len(d) > available_size:
            logger.debug(f">> {section.name}: truncating data ({len(d)} bytes) to {available_size} bytes")
            d = d[:available_size]
        # when section's data and raw size are zero, LIEF fails to append data to the target section, hence in this case
        #  we completely recreate the sections
        if l == section.size == 0:
            # save sections data
            sections, sections_data, l = sorted(list(parsed.sections), key=lambda s: s.virtual_address), [], len(d)
            for s in sections:
                sections_data.append({'name': ensure_str(s.name),
                                      'virtual_address': s.virtual_address,
                                      'char': s.characteristics,
                                      'content': list(d) if s == section else list(s.content),
                                      'size': l + [(fa - l % fa), 0][l % fa == 0] if s == section else s.size,
                                      'virtual_size': s.virtual_size,
                                      'modified': s == section})
            # remove all sections
            for s in sections:
                parsed.remove(s)
            # then recreate them with the updated section
            for sd in sections_data:
                new_sec = parsed.add_section(type(section)(content=sd['content'] + [0] * (-len(sd['content']) % fa),
                                                           name=sd['name'], characteristics=sd['char']))
                for k, v in sd.items():
                    if k in ["name", "char", "modified"]:
                        if k == "modified" and v:
                            section = new_sec
                        continue
                    setattr(new_sec, k, v)
            logger.debug(f"section: {new_sec.name} - content: {len(d)} - new raw size: {new_sec.size}")
        # classical case where section's data or raw size is not zero ; normally append data to the end of the section
        else:
            section.content = list(section.content) + d
            l = len(section.content)
            section.size = l + [(fa - l % fa), 0][l % fa == 0]
            logger.debug(f"section: {section.name} - content: {l} - new raw size: {section.size}")
    return _append_to_section



def move_entrypoint_to_new_section(name, section_type=None, characteristics=None, pre_data=b"", post_data=b"",
                                   with_dead_code=True):
    """ Set the entrypoint (EP) to a new section added to the binary that contains code to jump back to the original EP.
        The new section contains *pre_data*, then the code to jump back to the original EP, and finally *post_data*.
        (*code_data* is the code to be executed before the jump back to the original EP.)
    """
    _trampoline = lambda oep_or_offset: [0xE9, *list(oep_or_offset.to_bytes(4, "little", signed=oep_or_offset < 0))]
    @supported_parsers("lief") # 'parsed' is a lief.PE.Binary
    def _move_entrypoint_to_new_section(parsed, logger):
        from random import randint, sample
        logger.debug(f">> moving entrypoint to new section: {name}")
        oep, ep_data = parsed.optional_header.addressof_entrypoint, []
        # randomly add prolog (mostly found in NotPacked binaries)
        #   push ebp / mov ebp, esp /&/ sub esp, 0xc
        code_data = [[0x55, 0x8b, 0xec], []][randint(0, 1)] + [0x83, 0xec, 0x0c]
        # add dead code (dummy instructions that do nothing)
        if with_dead_code:
            dead_code, dcr = get_pe_data()["DEAD_CODE"], _DEAD_CODE_MAX_REPEAT
            # select k random elements from the dead_code list (With random number of repetitions for each element
            #  (allowing for repetition) )
            _dc_data = sample(dead_code, k=random.randint(*_DEAD_CODE_BOUNDS),
                              counts=[random.randint(dcr - (dcr // 2), dcr) for _ in range(len(dead_code))])
            _dc_data = ([x for l in _dc_data for x in l] if isinstance(_dc_data[0], list) else _dc_data) \
                       if isinstance(_dc_data, list) else _dc_data
            # restore stack (add esp, 0xc)
            code_data += [0x83, 0xc4, 0x0c]
            # add dead code to be executed before the jump back to the original EP
            ep_data += code_data
        # add trampoline code
        ep_data += _trampoline(oep)
        _characteristics = characteristics or parsed.SECTION_CHARACTERISTICS['MEM_READ'] | \
                                              parsed.SECTION_CHARACTERISTICS['MEM_EXECUTE']
        # create new section
        add_section(name, section_type or parsed.SECTION_TYPES['TEXT'], _characteristics,
                    list(pre_data) + ep_data + list(post_data))(parsed, logger)
        # update content and trampoline offset (do it after to know the address of the new section)
        s = parsed.get_section(name)
        offset = oep - (s.virtual_address + len(pre_data) + len(ep_data))
        s.content = list(pre_data) + code_data + _trampoline(offset) + list(post_data)
        # update EP
        parsed.optional_header.addressof_entrypoint = s.virtual_address + len(pre_data)
    return _move_entrypoint_to_new_section


def move_entrypoint_to_slack_space(section_input, pre_data=b"", post_data_source=b""):
    """ Set the entrypoint (EP) to a new section added to the binary that contains code to jump back to the original EP.
    """
    @supported_parsers("lief")
    def _move_entrypoint_to_slack_space(parsed, logger):
        if parsed.optional_header.section_alignment % parsed.optional_header.file_alignment != 0:
            raise ValueError("SectionAlignment is not a multiple of FileAlignment (file integrity cannot be assured)")
        address_bitsize = [64, 32]["32" in parsed.path.format]
        oep = parsed.optional_header.addressof_entrypoint + parsed.optional_header.imagebase
        #  push current_entrypoint
        #  ret
        ep_data = [0x68] + list(oep.to_bytes([4, 8][parsed.path.format[-2:] == "64"], 'little')) + [0xc3]
        # other possibility:
        #  mov eax current_entrypoint
        #  jmp eax
        #  ep_data = [0xb8] + list(oep.to_bytes(4, 'little')) + [0xff, 0xe0]
        d = list(pre_data) + ep_data
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


def set_checksum(value):
    """ Set the checksum. """
    @supported_parsers("lief")
    def _set_checksum(parsed, logger):
        parsed.optional_header.checksum = value
    return _set_checksum

