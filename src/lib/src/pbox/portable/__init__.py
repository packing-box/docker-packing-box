# -*- coding: UTF-8 -*-
""" Portable code: parsers and primitives with no dependency outside the standard library and no import from the rest
     of pbox, so that it can be copied as is into the standalone tools exported from trained models.

pbox uses it too (as the 'portable' parser, see pbox.core.executable.parsers.portable), so that the values computed
 by an exported tool and by the Packing Box come from the same code.
"""
