import lief

__all__ = ["lief_parser"]

def lief_parser(modifier):
    def _wrapper(d, executable=None, **kw):
        parsed = lief.parse(str(executable.destination))
        kw.update(parsed=parsed)
        d.update(sections=parsed.sections)
        modifier(d, **kw)
        builder = lief.PE.Builder(parsed)
        #builder.build_imports() 
        #TODO: build imports only when necessary, because it adds a .l1 section
        builder.build()
        builder.write(str(executable.destination))
    return _wrapper