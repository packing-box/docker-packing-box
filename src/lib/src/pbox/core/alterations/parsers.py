# -*- coding: UTF-8 -*-
from tinyscript.helpers import lazy_load_module

lazy_load_module("lief")


__all__ = ["parse_executable", "parse_section", "parser_handler"]
__parsers__ = ["lief_parser"]


class SectionAbstract:
    """ Class to simplify the handling of sections.
    
    NB: Simple namespace to hold section information. Referencing a lief.PE.Section directly is dangerous, because it
         can be modified by alterations, which will break things. Also, if different parsers are used in susequent
         alterations, a common format is required.
    """
    __slots__ = ["name", "virtual_size", "size", "offset", "content", "virtual_address"]
    
    def __init__(self, section):
        self.name = section.name
        self.virtual_size = section.virtual_size
        self.size = section.size
        self.offset = section.offset
        self.content = list(section.content)
        self.virtual_address = section.virtual_address


def parse_executable(parser, exe, namespace):
    """ Updates the namespace with information from the binary. If no parser is specified, lief.parse is used.
    
    :param parser: parser to use
    :param exe:    executable (pbox.[common|learning].Executable instance) to extract information from
    :param d:      namespace (as a dictionary) for modfifier evaluation
    """
    if parser is None:
        parser = lief_parser(exe)
    namespace['compute_checksum'] = parser.compute_checksum
    namespace['sections'] = parser.get_sections()


def parse_section(section, parsed):
    """ Helper for allowing to use sections or section names as input. """
    print("[+]", section)
    if isinstance(section, SectionAbstract) or isinstance(section, lief.PE.Section):
        return next(s for s in parsed.sections if s.name == section.name and \
                                                  s.virtual_address == section.virtual_address)
    elif isinstance(section, str):
        print("section is of type str")
        print(parsed.get_section(section))
        return parsed.get_section(section)
    else:
        raise TypeError("Section must be lief.PE.Section, SectionAbstract or str, not %s" % section.__class__)


def parser_handler(parser_name):
    """ Decorator for applying a parser.

    :param parser: parser's name as a string
    """
    def decorator(alteration_wrapper):
        def wrapper(parser=None, executable=None, parsed=None, **kw):
            # parsed is not None only for testing, otherwise, it should be contained in a lief_parser object
            if parsed is not None:
                alteration_wrapper(parsed=parsed, **kw)
                return None
            # redefine parser if necessary
            if parser_name not in __parsers__:
                raise ValueError("Parser '%s' could not be found" % parser_name)
            # if None was used
            if parser is None:
                parser = globals()[parser_name](executable)
            # if another was used for previous alterations, close the previous one
            elif parser.__class__.__name__ != parser_name:
                parser.build()
                parser = globals()[parser_name](executable)
            # call the alteration wrapper via the parser
            parser(alteration_wrapper, executable=executable, **kw)
            return parser
        return wrapper
    return decorator


# -------------------------------------------------- Parsers -----------------------------------------------------------
class lief_parser:
    name = "lief_parser"
    
    def __init__(self, executable):
        self.executable = executable
        self.parsed = lief.parse(str(self.executable.destination))
        self.build_instructions = {"imports": None, "dos_stub": None, "patch_imports": None}
        self.overlay = self.parsed.overlay
    
    def __call__(self, modifier, **kw):
        """ Calls the modifier function with the parsed executable. """
        kw.update(parsed=self.parsed, overlay=self.overlay)
        out = modifier(**kw)
        if out is not None:
            for i in out:
                if self.build_instructions[i] is None:
                    self.build_instructions[i] = out[i]
                else:
                    self.build_instructions[i] &= out[i]
    
    def build(self):        
        bi = self.build_instructions.copy()
        for i in bi:
            if bi[i] is None:
                bi[i] = False
        builder = lief.PE.Builder(self.parsed)
        builder.build_imports(self.build_instructions["imports"])
        builder.patch_imports(self.build_instructions["patch_imports"])
        builder.build_overlay(False)  # build_overlay(True) fails when adding a section to the binary
        builder.build()
        builder.write(str(self.executable.destination))
        with open(str(self.executable.destination), 'ab') as f:
            f.write(bytes(self.overlay))
    
    def compute_checksum(self):
        self.build()
        self.__init__(self.executable)
        return self.parsed.optional_header.computed_checksum
    
    def get_sections(self):
        return [SectionAbstract(s) for s in self.parsed.sections]

