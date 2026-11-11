# -*- coding: UTF-8 -*-
""" Detection of the executable formats pbox distinguishes for PE files ("PE32", "PE64", ".NET", "MSDOS"), from the
     rules of libmagic (file 5.46, magic/Magdir/msdos and archive) that produce the descriptions pbox matches (see
     SIGNATURES in pbox.constants). """
import struct


__all__ = ["detect"]

BYTES_MAX = 7 * 1024 * 1024   # libmagic 5.46's default


def detect(data):
    """ Format of an executable given its bytes, or None if pbox would not consider it as an executable. """
    # ZIP archive with prepended data (e.g. self-extracting archives): libmagic describes them as such, not as PE ;
    #  NB: libmagic only reads the first 7 MiB of a file (bytes_max), the offsets from the end are relative to them
    buf = data[:BYTES_MAX]
    if len(buf) >= 22 and buf[-2:] == b"\0\0" and buf[-22:-18] == b"PK\x05\x06" and not buf.startswith(b"#!"):
        return
    if len(data) < 64 or data[:2] != b"MZ":
        return
    nt = struct.unpack_from("<I", data, 0x3c)[0]
    if data[nt:nt+4] != b"PE\0\0" or nt + 26 > len(data):
        return "MSDOS"
    magic = struct.unpack_from("<H", data, nt + 24)[0]
    if magic not in (0x10b, 0x20b):
        return "MSDOS"
    # COM runtime descriptor: NumberOfRvaAndSizes (as a signed 16-bit value) > 14 and data directory 14's RVA (as a
    #  signed 32-bit value) > 0
    n_dirs, clr = (nt + 132, nt + 248) if magic == 0x20b else (nt + 116, nt + 232)
    if n_dirs + 2 <= len(data) and struct.unpack_from("<h", data, n_dirs)[0] > 14 and \
       clr + 4 <= len(data) and struct.unpack_from("<i", data, clr)[0] > 0:
        return ".NET"
    return "PE64" if magic == 0x20b else "PE32"
