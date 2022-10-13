# -*- coding: UTF-8 -*-
from ast import literal_eval
from tinyscript.helpers import execute_and_log as run


__all__ = ["pefeats", "STD_SECTIONS"]


# This list is computed taking binaries into account from:
#  - ~/.wine/drive_c/windows/system32, ~/.wine/drive_c/windows/syswow64 (Wine)
#  - https://github.com/packing-box/dataset-packed-pe/tree/master/not-packed (set of standard not packed PE files)
#  - https://github.com/roussieau/masterthesis/blob/master/src/detector/tools/pefeats/pefeats.cpp (hardcoded list)
STD_SECTIONS = """
.00cfg
.CRT
.bss
.buildid
.cormeta
.data
.debug
.debug$F
.debug$P
.debug$S
.debug$T
.didata
.drective
.edata
.eh_fram
.gfids
.idata
.idlsym
.itext
.ndata
.pdata
.qtmetad
.rdata
.reloc
.rsrc
.sbss
.sdata
.shared
.srdata
.sxdata
.text
.tls
.tls$
.vsdata
.xdata
BSS
CODE
DATA
code
const
data
""".strip().split("\n")


def pefeats(executable):
    """ This uses pefeats to extract 119 features from PE files. """
    out, err, retc = run("pefeats \'%s\'" % executable)
    if retc == 0:
        values = []
        for x in out.decode().strip().split(",")[1:]:
            try:
                values.append(literal_eval(x))
            except ValueError:
                values.append(x)
        return values

