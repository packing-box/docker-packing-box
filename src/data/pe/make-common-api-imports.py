#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from tinyscript import *


def compose_common_apis():
    COMMON_IMPORTS = [
        'GetStdHandle', 'ZwProtectVirtualMemory', 'RtlFormatCurrentUserKeyPath', 'EnterCriticalSection',
        'WideCharToMultiByte', 'RaiseException', 'RegCloseKey', 'GetStartupInfo', 'GetModuleFileName', 'RtlUnwind',
        'CreateFile', 'LoadLibrary', 'InitializeCriticalSection', 'MessageBox', 'GetModuleHandle', '__p__acmdln',
        'GetProcAddress', 'SysFreeString', 'PrintDlg', 'ExitProcess', 'VerQueryValue', 'WriteFile', 'LocalAlloc',
        'FreeLibrary', 'RegQueryValue', 'RegOpenKey', 'VirtualProtect', 'VirtualQuery', 'SetThreadContext',
        'LdrLoadDll', 'SHGetFolderPath', 'VirtualAlloc', 'GetPrivateProfileSection', 'LocalFree', 'GetCommandLine',
        'LdrGetProcedureAddress'
    ]
    return {f+s for f in COMMON_IMPORTS for s in ('', 'A', 'W', 'Ex', 'ExA', 'ExW')}


def filter_out_special_apis(apis):
    PATTERNS = [r"^\?", "^ord\d+", r"^_", r"^\$"]
    for p in PATTERNS:
        apis = {a for a in apis if not re.search(p, a)}
    return apis


if __name__ == '__main__':
    parser.add_argument("-s", "--source", default="common_dll_imports.json", type=ts.file_exists,
                        help="source JSON file for common DLL imports")
    initialize()
    with open(args.source) as fin, open("common_api_imports.txt", 'wt') as fout:
        cai = set(functools.reduce(lambda i, j: i+j, json.load(fin).values() or [()]))
        l = filter_out_special_apis(sorted(cai | compose_common_apis()))
        fout.write("\n".join(sorted(l, key=lambda x: x.lower())))

