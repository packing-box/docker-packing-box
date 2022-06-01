# -*- coding: UTF-8 -*-
is_equal  = lambda v: lambda x: x == v
is_in     = lambda *v: lambda x: x in v
is_get    = lambda f: lambda x: x >= f
is_let    = lambda f: lambda x: x <= f
is_mult   = lambda i: lambda x: x % i == 0
is_not_in = lambda *v: lambda x: x not in v
is_not_0  = lambda x: x != 0
is_0      = lambda x: x == 0
is_within = lambda v1, v2: lambda x: v1 <= x <= v2


# combinations only support exact feature names
FEATURE_TRANSFORMERS = {
    # default transformers are applied, whatever the chosen transformation class
    'default': {
        r'^base_of_code$': (
            lambda x: -1 if x % 0x1000 != 0 else x / 0x1000,
            "Base of Code (reduced to its multiplier of 0x1000 ; -1 means it is not a multiplier)"
        ),
        # see: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format
        #  Windows CE: 0x00010000 ; other Windows: 0x00400000 ; DLL: 0x10000000
        r'^image_base$': {
            'is_image_base_standard': (
                is_in(0x10000, 0x400000, 0x10000000),
                "Is the ImageBase standard (Windows CE: 0x10000, other Windows PE: 0x400000, DLL: 0x10000000)"
            ),
        },
    },
    'boolean': {
        r'^base_of_code$': {
            'is_base_of_code_standard': (
                is_mult(0x1000),
                "Is the BaseOfCode a multiplier of 0x1000"
            ),
        },
        r'^checksum$': {
            'is_checksum_null': (
                is_0,
                "Is the checksum set to 0"
            ),
        },
        r'^os_major_version$': {
            'is_os_major_version_4': (
                is_equal(4),
                "Is the OS major version equal to 4"
            ),
            'is_os_major_version_5': (
                is_equal(5),
                "Is the OS major version equal to 5"
            ),
            'is_os_major_version_6': (
                is_equal(6),
                "Is the OS major version equal to 6"
            ),
            'is_os_major_version_standard': (
                is_within(4, 10),
                "Is the OS major version within 4 (<Windows XP) and 10 (Windows 11)"
            ),
        },
        r'^os_minor_version$': {
            'is_os_minor_version_4': (
                is_equal(0),
                "Is the OS minor version equal to 0"
            ),
            'is_os_minor_version_5': (
                is_equal(1),
                "Is the OS minor version equal to 1"
            ),
            'is_os_minor_version_6': (
                is_equal(2),
                "Is the OS minor version equal to 2"
            ),
            'is_os_minor_version_6': (
                is_equal(3),
                "Is the OS minor version equal to 3"
            ),
            'is_os_minor_version_standard': (
                is_within(0, 3),
                "Is the OS minor version within 0 and 3"
            ),
        },
        ("size_of_code", "size_of_image"): {
            'is_code_section_90%_of_image_size': (
                lambda x, y: x / y >= .9,
                "Is the size of the code section greather or equal to 90% of the image size"
            ),
        },
        r'^size_of_headers$': {
            'is_size_of_headers_512B': (
                is_equal(512),
                "Is the size of the headers equal to 512B"
            ),
            'is_size_of_headers_1024B': (
                is_equal(1024),
                "Is the size of the headers equal to 1024B"
            ),
            'is_size_of_headers_1536B': (
                is_equal(1536),
                "Is the size of the headers equal to 1536B"
            ),
            'is_size_of_headers_4096B': (
                is_equal(4096),
                "Is the size of the headers equal to 4096B"
            ),
            'is_size_of_headers_non_standard': (
                is_not_in(512, 1024, 1536, 4096),
                "Is the size of the headers not equal to any of 512B, 1024B, 1536B or 4096B"
            ),
        },
        r'^size_of_initializeddata$': {
            'is_size_of_initializeddata_notnull': (
                is_not_0,
                "Is the size of initialized data not null"
            ),
            'is_size_of_initializeddata_get_1MB': (
                is_get(1024*1024),
                "Is the size of initialized data greater than or equal to 1MB"
            ),
            'is_size_of_initializeddata_get_2MB': (
                is_get(2*1024*1024),
                "Is the size of initialized data greater than or equal to 2MB"
            ),
            'is_size_of_initializeddata_get_3MB': (
                is_get(3*1024*1024),
                "Is the size of initialized data greater than or equal to 3MB"
            ),
            'is_size_of_initializeddata_get_4MB': (
                is_get(4*1024*1024),
                "Is the size of initialized data greater than or equal to 4MB"
            ),
            'is_size_of_initializeddata_get_5MB': (
                is_get(5*1024*1024),
                "Is the size of initialized data greater than or equal to 5MB"
            ),
        },
        r'^size_of_initializeddata$': {
            'is_size_of_uninitializeddata_notnull': (
                is_not_0,
                "Is the size of uninitialized data not null"
            ),
            'is_size_of_uninitializeddata_get_1MB': (
                is_get(1024*1024),
                "Is the size of uninitialized data greater than or equal to 1MB"
            ),
            'is_size_of_uninitializeddata_get_2MB': (
                is_get(2*1024*1024),
                "Is the size of uninitialized data greater than or equal to 2MB"
            ),
            'is_size_of_uninitializeddata_get_3MB': (
                is_get(3*1024*1024),
                "Is the size of uninitialized data greater than or equal to 3MB"
            ),
            'is_size_of_uninitializeddata_get_4MB': (
                is_get(4*1024*1024),
                "Is the size of uninitialized data greater than or equal to 4MB"
            ),
            'is_size_of_uninitializeddata_get_5MB': (
                is_get(5*1024*1024),
                "Is the size of uninitialized data greater than or equal to 5MB"
            ),
        },
    },
}

