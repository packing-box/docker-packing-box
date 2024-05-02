# -*- coding: UTF-8 -*-
from .__common__ import *
from .__common__ import __all__


class FeatureExtractionFunctions:

    @staticmethod
    def get_api_features(cfg, mal_apis, non_apis={'PathTerminator'}, suffixes={'', 'Ex', 'A', 'ExA'}):
        """Extracts features based on api imports and use

        Args:
            cfg: a CFG object
            mal_apis: a list of strings of malicious api names
        KwArgs:
            non_apis: a set of Angr zero-sized node names that are non-apis
            suffixes: a set of suffixed that can be appended to mal_apis, resulting in (possible) similar implementations of those apis
        Returns:
            api_uniqmalused_uniqtotused_ratio: ratio of (unique malicious api functions used) / (unique api functions used)
            api_malused_totused_ratio: ratio of (total malicious api functions used) / (total api functions used) [Note: doesn't take into account that loops can cause some of these nodes to be traversed more than others]
            api_uniqmalused_totimp_ratio: ratio of (unique malicious api functions used) / (api functions imported)
            api_malimp_totimp_ratio: ratio of (malicious api functions imported) / (api functions imported) [also implemented in pefeats, but with fixed set of mal_apis, while here it is a function parameter]
            api_malapis_present: list of booleans indicating whether the mal_api (or a similar version with a suffix) is imported or not
        """
        imported_apis = cfg.model.project.loader.main_object.imports.keys()
        if suffixes:
            stripped_apis = []
            non_empty_suffixes = {s for s in suffixes if s != ''}
            for api in imported_apis:
                for suffix in non_empty_suffixes:
                    if api.endswith(suffix):
                        api = api[:-len(suffix)]
                        break
                stripped_apis.append(api)
            imported_apis = stripped_apis
        mal_apis_set = set(mal_apis)
        mal_imported_apis = [api for api in imported_apis if api in mal_apis_set]
        used_apis = [n.name for n in cfg.model.nodes() if n.name is not None and n.name not in non_apis]
        unique_used_apis = set(used_apis)
        mal_used_apis = [api for api in used_apis if api in mal_apis_set]
        unique_mal_used_apis = set(mal_used_apis)
        api_uniqmalused_uniqtotused_ratio = api_malused_totused_ratio = -1 # Default value if no used apis
        api_uniqmalused_totimp_ratio = api_malimp_totimp_ratio = -1 # Default value if no imported_apis
        if unique_used_apis:
            api_uniqmalused_uniqtotused_ratio = len(unique_mal_used_apis) / len(unique_used_apis)
            api_malused_totused_ratio = len(mal_used_apis) / len(used_apis)
        if imported_apis:
            api_uniqmalused_totimp_ratio = len(unique_mal_used_apis) / len(imported_apis)
            api_malimp_totimp_ratio = len(mal_imported_apis) / len(imported_apis)
        api_malapis_present = [api in imported_apis for api in mal_apis]
        return api_uniqmalused_uniqtotused_ratio, api_malused_totused_ratio, api_uniqmalused_totimp_ratio, api_malimp_totimp_ratio, api_malapis_present

    @staticmethod
    def get_instruction_string_features(cfg):
        """Extracts features based on instruction strings
        
        Args:
            cfg: a CFG object
        Returns:
            ratio_indir_jmps: ratio of indirect jump-like instructions (jumps or calls) over the total number of jump-like instructions
            ratio_register_type_counts_over_total_register_counts: a list of ratios of (number of occurences of a set of predefined register types in the operand strings of the instructions in the nodes in all subgraphs) / (total number of register occurences in the extracted operand strings)
            ratio_register_type_counts_over_num_instructions: a list of ratios of (number of occurences of a set of predefined register types in the operand strings of the instructions in the nodes in all subgraphs) / (total number extracted instructions)
        """
        jump_mnemonics = {"call", "jmp", "bnd jmp", "je", "jne", "jz", "jnz", "ja", "jae", "jb", "jbe", "jl", "jle", "jg", "jge", "jo", "jno", "js", "jns", "jp", "jnp", "jecxz", "jrcxz", "jmpf", "jmpq", "jmpw"}
        num_jump_insns = num_indirect_jump_insns = 0
        for mnemonic, op_str in cfg.iterinsns():
            if mnemonic in jump_mnemonics:
                num_jump_insns += 1
                if op_str.startswith('0x'):
                    num_indirect_jump_insns += 1
        groups = {
            "Return value": {"rax", "eax", "ax", "ah", "al"},
            "General-Purpose Registers": {"rbx", "rcx", "rdx", "ebx", "ecx", "edx", "bx", "bh", "bl", "cx", "ch", "cl", "dx", "dh", "dl"},
            "Segment Registers": {"cs", "ds", "es", "fs", "gs", "ss"},
            "Function arguments": {"rsi", "rdi", "r8", "r9", "esi", "edi"},
            "Stack Registers": {"rbp", "rsp", "ebp", "esp"},
            "Instruction Pointer": {"rip", "eip"},
            "Flags Register": {"rflags", "eflags"},
            "Floating-Point Registers": set("xmm{}".format(i) for i in range(16))
        }
        register_count = {group: 0 for group in groups}
        re.sub(r"[,\+\*\[\]]", " ", )
        trans_table = str.maketrans({',': ' ', '+': ' ', '*': ' ', '[': ' ', ']': ' ', '-': ' '})
        operand_tokens = sum((it[1].translate(trans_table).split() for it in cfg.iterinsns()), [])
        for token in operand_tokens:
            for group, register_set in groups.items():
                if token in register_set:
                    register_count[group] += 1
        register_type_counts = list(register_count.values())
        ratio_indir_jmps = -1 # Default value if num_jump_insns is zero
        if num_jump_insns:
            ratio_indir_jmps = num_indirect_jump_insns / num_jump_insns
        return ratio_indir_jmps, \
               [-1 if sum(register_type_counts) == 0 else count / sum(register_type_counts) for count in register_type_counts], \
               [-1 if len(insn_tuples) == 0 else count / len(insn_tuples) for count in register_type_counts]


