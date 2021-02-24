# -*- coding: UTF-8 -*-
from functools import cached_property
from tinyscript.helpers import execute_and_log as run

from ..utils import expand_categories


__all__ = ["Features", "FEATURE_DESCRIPTIONS"]


FEATURE_DESCRIPTIONS = {}
FEATURES = {
    'All': {
        'checksum': lambda: 1,
    },
}


FEATURE_DESCRIPTIONS = {
    'dll_characteristics_1': "DLLs characteristics 1",
    'dll_characteristics_2': "DLLs characteristics 2",
    'dll_characteristics_3': "DLLs characteristics 3",
    'dll_characteristics_4': "DLLs characteristics 4",
    'dll_characteristics_5': "DLLs characteristics 5",
    'dll_characteristics_6': "DLLs characteristics 6",
    'dll_characteristics_7': "DLLs characteristics 7",
    'dll_characteristics_8': "DLLs characteristics 8",
    'checksum': "Checksum",
    'image_base': "Image Base",
    'base_of_code': "Base of Code",
    'os_major_version': "OS Major version",
    'os_minor_version': "OS Minor version",
    'size_of_image': "Size of Image",
    'size_of_code': "Size of Code",
    'headers': "Headers",
    'size_of_initializeddata': "Size Of InitializedData",
    'size_of_uninitializeddata': "Size Of UninitializedData",
    'size_of_stackreverse': "Size Of StackReserve",
    'size_of_stack_commit': "Size of Stack Commit",
    'section_alignment': "Section Alignment",
    'number_standard_sections': "number of standards sections the PE holds",
    'number_non_standard_sections': "number of non-standards sections the PE holds",
    'ratio_standard_sections': "ratio between the number of standards sections found and the number of all sections "
                               "found in the PE under analysis",
    'number_x_sections': "number of Executable sections the PE holds",
    'number_w_sections': "number of Writable sections the PE holds",
    'number_wx_sections': "number of Writable and Executable sections the PE holds",
    'number_rx_sections': "number of readable and executable sections",
    'number_rw_sections': "number of readable and writable sections",
    'number_rwx_sections': "number of Writable and Readable and Executable sections the PE holds",
    'code_section_x': "code section is not executable",
    'x_section_is_not_code': "executable section is not a code section",
    'code_section_not_present': "code section is not present in the PE under analysis",
    'ep_not_in_code_section': "EP is not in the code section",
    'ep_not_in_standard_section': "EP is not in a standard section",
    'ep_not_x_section': "EP is not in an executable section",
    'ep_ratio': "EP ratio between raw data and virtual size for the section of entry point",
    'number_sections_size_0': "number of sections having their physical size =0 (size on disk)",
    'number_sections_vsize>dsize': "number of sections having their virtual size greater than their raw data size",
    'max_ratio_rdata': "maximum ratio raw data per virtual size among all the sections",
    'min_ratio_rdata': "minimum ratio raw data per virtual size among all the sections",
    'address_to_rdata_not_conform': "address pointing to raw data on disk is not conforming with the file alignement",
    'entropy_code_sections': "entropy of Code/text sections",
    'entropy_data_section': "entropy of data section",
    'entropy_resource_section': "entropy of resource section",
    'entropy_pe_header': "entropy of PE header",
    'entropy': "entropy of the entire PE file",
    'entropy_section_with_ep': "entropy of section holding the Entry point (EP) of the PE under analysis",
    'byte_0_after_ep': "byte 0 following the EP",
    'byte_1_after_ep': "byte 1 following the EP",
    'byte_2_after_ep': "byte 2 following the EP",
    'byte_3_after_ep': "byte 3 following the EP",
    'byte_4_after_ep': "byte 4 following the EP",
    'byte_5_after_ep': "byte 5 following the EP",
    'byte_6_after_ep': "byte 6 following the EP",
    'byte_7_after_ep': "byte 7 following the EP",
    'byte_8_after_ep': "byte 8 following the EP",
    'byte_9_after_ep': "byte 9 following the EP",
    'byte_10_after_ep': "byte 10 following the EP",
    'byte_11_after_ep': "byte 11 following the EP",
    'byte_12_after_ep': "byte 12 following the EP",
    'byte_13_after_ep': "byte 13 following the EP",
    'byte_14_after_ep': "byte 14 following the EP",
    'byte_15_after_ep': "byte 15 following the EP",
    'byte_16_after_ep': "byte 16 following the EP",
    'byte_17_after_ep': "byte 17 following the EP",
    'byte_18_after_ep': "byte 18 following the EP",
    'byte_19_after_ep': "byte 19 following the EP",
    'byte_20_after_ep': "byte 20 following the EP",
    'byte_21_after_ep': "byte 21 following the EP",
    'byte_22_after_ep': "byte 22 following the EP",
    'byte_23_after_ep': "byte 23 following the EP",
    'byte_24_after_ep': "byte 24 following the EP",
    'byte_25_after_ep': "byte 25 following the EP",
    'byte_26_after_ep': "byte 26 following the EP",
    'byte_27_after_ep': "byte 27 following the EP",
    'byte_28_after_ep': "byte 28 following the EP",
    'byte_29_after_ep': "byte 29 following the EP",
    'byte_30_after_ep': "byte 30 following the EP",
    'byte_31_after_ep': "byte 31 following the EP",
    'byte_32_after_ep': "byte 32 following the EP",
    'byte_33_after_ep': "byte 33 following the EP",
    'byte_34_after_ep': "byte 34 following the EP",
    'byte_35_after_ep': "byte 35 following the EP",
    'byte_36_after_ep': "byte 36 following the EP",
    'byte_37_after_ep': "byte 37 following the EP",
    'byte_38_after_ep': "byte 38 following the EP",
    'byte_39_after_ep': "byte 39 following the EP",
    'byte_40_after_ep': "byte 40 following the EP",
    'byte_41_after_ep': "byte 41 following the EP",
    'byte_42_after_ep': "byte 42 following the EP",
    'byte_43_after_ep': "byte 43 following the EP",
    'byte_44_after_ep': "byte 44 following the EP",
    'byte_45_after_ep': "byte 45 following the EP",
    'byte_46_after_ep': "byte 46 following the EP",
    'byte_47_after_ep': "byte 47 following the EP",
    'byte_48_after_ep': "byte 48 following the EP",
    'byte_49_after_ep': "byte 49 following the EP",
    'byte_50_after_ep': "byte 50 following the EP",
    'byte_51_after_ep': "byte 51 following the EP",
    'byte_52_after_ep': "byte 52 following the EP",
    'byte_53_after_ep': "byte 53 following the EP",
    'byte_54_after_ep': "byte 54 following the EP",
    'byte_55_after_ep': "byte 55 following the EP",
    'byte_56_after_ep': "byte 56 following the EP",
    'byte_57_after_ep': "byte 57 following the EP",
    'byte_58_after_ep': "byte 58 following the EP",
    'byte_59_after_ep': "byte 59 following the EP",
    'byte_60_after_ep': "byte 60 following the EP",
    'byte_61_after_ep': "byte 61 following the EP",
    'byte_62_after_ep': "byte 62 following the EP",
    'byte_63_after_ep': "byte 63 following the EP",
    'number_dll_imported': "number of DLLs imported",
    'number_func_imported_in_idt': "number of functions imported found in the import table directory (IDT)",
    'number_malicious_api_imported': "number of malicious APIs imported",
    'ratio_malicious_api_imported': "ratio between the number of malicious APIs imported to the number of all functions"
                                    " imported by the PE",
    'number_addresses_in_iat': "number of addresses (corresponds to functions) found in the import address table (IAT)",
    'debug_dir_present': "debug directory is present or not",
    'number_resources': "number of resources the PE holds",
}


def pefeats(executable):
    """ This uses pefeats to extract 119 features from PE files. """
    out, err, retc = run("pefeats %s" % executable)
    if retc == 0:
        out = None


class Features(dict):
    """ This class represents the dictionary of features valid for a given list of executable categories. """
    def __init__(self, *categories):
        categories, all_categories = expand_categories(*categories), expand_categories("All")
        # consider most generic features first
        for category, features in FEATURES.items():
            if category in all_categories:
                continue
            for subcategory in expand_categories(category):
                if subcategory in categories:
                    for name, func in features.items():
                        self[name] = func
        # then consider most specific ones
        for category, features in FEATURES.items():
            if category not in all_categories or category not in categories:
                continue
            for name, func in features.items():
                self[name] = func
    
    @cached_property
    def descriptions(self):
        """ Dictionary of feature names and their descriptions applicable given the input categories. """
        return {n: FEATURE_DESCRIPTIONS.get(n, "") for n in self.keys()}

