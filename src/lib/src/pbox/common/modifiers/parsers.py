import lief

__all__ = ["parser_handler", "parse_exe_info_default", "SectionAbstract"]
__parsers__ = ["lief_parser"]


def parser_handler(parser_name):
    """Wrapper to apply a parser

    Args:
        parser_name (str): parser to apply
    """
    def decorator(modifier_func):
        def wrapper(parser=None, executable=None, parsed=None, **kw):
            # parsed is not None only for testing, otherwise, it should be contained in a lief_parser object
            if parsed is not None:
                modifier_func(parsed=parsed ,**kw)
                return None
            
            # redefine parser if necessary
            if parser_name not in __parsers__:
                raise ValueError("Parser {parser_name} could not be found")

            # if none was used
            if parser is None:
                parser = globals()[parser_name](executable)

            # if another was used for previous modifiers, close the previous one
            elif parser.__class__.__name__ != parser_name:
                parser.build()
                parser = globals()[parser_name](executable)

            # call the modifier via the parser
            parser(modifier_func, executable=executable, **kw)
            return parser
        return wrapper
    return decorator


class lief_parser():
    name = "lief_parser"
    def __init__(self, executable):
        self.executable = executable
        self.parsed = lief.parse(str(self.executable.destination))
        self.build_instructions = {"imports": None, "dos_stub": None, "patch_imports":None}
        self.overlay = self.parsed.overlay
        
    def __call__(self, modifier_func, **kw):
        """Calls the modifier function with the parsed executable

        Args:
            modifier_func (function): Modifier function
        """
        kw.update(parsed=self.parsed, overlay=self.overlay)
        out = modifier_func(**kw)
        if out is not None:
            for i in out:
                if self.build_instructions[i] is None:
                    self.build_instructions[i] = out[i]
                else:
                    self.build_instructions[i] &= out[i]

    def get_sections(self):
        return [SectionAbstract(s) for s in self.parsed.sections]
    
    def compute_checksum(self):
        self.build()
        self.__init__(self.executable)
        return self.parsed.optional_header.computed_checksum

    def build(self):        
        bi = self.build_instructions.copy()
        for i in bi:
            if bi[i] is None:
                bi[i] = False
        
        builder = lief.PE.Builder(self.parsed)
        builder.build_imports(self.build_instructions["imports"])
        builder.patch_imports(self.build_instructions["patch_imports"])
        builder.build_overlay(False) # build_overlay(True) fails when adding a section to the binary
        builder.build()
        
        builder.write(str(self.executable.destination))
        with open(str(self.executable.destination), 'ab') as f:
            f.write(bytes(self.overlay))


class SectionAbstract():
    """Object to simplify the handling of sections
        Simple namespace to hold section information. Referencing a lief.PE.Section directly is dangerous, 
        because it can be modified by alterations, which will break things.
        Also, if different parsers are used in susequent alterations, a common format is required.
    """
    def __init__(self, section):
        self.name = section.name
        self.virtual_size = section.virtual_size
        self.size = section.size
        self.offset = section.offset
        self.content = list(section.content)
        self.virtual_address = section.virtual_address
            

def parse_exe_info_default(parser, exe, d):
    """Updates the namespace with information from the binary. If no parser is specified, lief.parse is used.

    Args:
        parser (pbox.lief_parser): parser to use
        exe (pbox.Executable): executable to extract information from
        d (dict): namespace for modfifier evaluation 

    Returns:
        _type_: _description_
    """
    if parser is None:
        tmp_parser = lief.parse(str(exe.destination))
        d.update(sections=[SectionAbstract(s) for s in tmp_parser.sections],
                compute_checksum=lambda _:tmp_parser.optional_header.computed_checksum)
    else:
        d.update(sections=parser.get_sections(),
                compute_checksum=parser.compute_checksum)
    return d

