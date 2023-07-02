# -*- coding: UTF-8 -*-
import lief
from random import choices
import os.path as osp
import json

__all__ = ["COMMON_PE_SECTION_NAMES", "STD_SECTION_NAMES", "COMMON_PACKER_SECTION_NAMES", "COMMON_API_IMPORTS", "choices", "next"]

path = osp.dirname(__file__)

with open(osp.join(path, "common_section_names_PE.txt")) as fin:
    COMMON_PE_SECTION_NAMES = [l for l in fin.readlines() if not l.startswith('#')]

with open(osp.join(path, "common_section_names_packers.txt")) as fin:
    COMMON_PACKER_SECTION_NAMES = [l for l in fin.readlines() if not l.startswith('#')]

with open(osp.join(path, "common_dll_imports.json")) as fin:
    d = json.load(fin)
    COMMON_API_IMPORTS = [(lib, api) for lib in d for api in d[lib]]

STD_SECTION_NAMES = ['.text','.data','.crt','.bss','.rsrc','.idata','.edata','.reloc','.tls','.rdata']

next=next
