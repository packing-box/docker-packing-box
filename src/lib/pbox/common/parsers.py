import lief

__all__ = ["lief_parser"]

def lief_parser(modifier):
    def _wrapper(d, executable=None, **kw):
        parsed = lief.parse(executable.realpath)
        kw.update(parsed=parsed)
        d.update(sections=parsed.sections)
        modifier(d, **kw)
        builder = lief.PE.Builder(parsed)
        builder.build_imports(True)
        builder.build()
        builder.write(executable.realpath)
    return _wrapper