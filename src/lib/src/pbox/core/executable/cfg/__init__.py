# -*- coding: UTF-8 -*-
from .__common__ import *
from .__common__ import __all__


class FeatureExtractionFunctions:

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


