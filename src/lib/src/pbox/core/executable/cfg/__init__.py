# -*- coding: UTF-8 -*-
from .__common__ import *
from .__common__ import __all__



def order_by_occurence_then_value(vector):
    """Orders the elements of the input list first by occurence, then by value
    
    Args:
        vector: a list of unordered elements
    Returns:
        values: a list of ordered elements, first by occurence, then by value
        counts: a list of number of occurences of the elements in 'values'
    """
    from collections import Counter
    counts = Counter(vector) # This line can be replaced with the commented code below, in case you prefer no extra import. I currently kept it as it's twice as fast as the commented code below
    #counts = {}
    #for item in vector:
    #    counts[item] = counts.get(item, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    values = [value for value, count in ordered]
    counts = [count for value, count in ordered]
    return values, counts


class FeatureExtractionFunctions:

    @staticmethod
    def get_ngram_dist_features(cfg, n, len_dist, len_vals, len_zeros, across_nodes=True):
        """Extracts features based on ngrams, taking all nodes in the root subgraph into account
        
        Args:
            cfg: a CFG object
            n: the number of bytes in the ngram
            len_dist: the length of the ngram distribution count feature vector
            len_vals: the length of the feature vector formed by taking the first len_vals bytes of the sorted (with order_by_occurence_then_value(vector)) byte strings in the ngrams_list
            len_zeros: the amount of first bytes to take into account of the sorted (with order_by_occurence_then_value(vector)) byte strings in the ngrams_list when counting the number of zero-bytes among them
        KwArgs:
            across_nodes: whether to use the across_nodes method or the zeropad method for ngram extraction
        Returns:
            ngram_dist: a list representing the distribution of extracted ngrams from the graph downwards connected to the (sub_)root_node, using the zeropad or across_nodes method. The list is sorted from most number of occurences to least and contains the number of occurences of the first len_dist ngrams
            ngram_vals: a bytestring composed of the first len_vals most occuring ngrams, concatenated
            ngram_len: the original amount of different extracted ngrams, before fixing them to a specific length (with zeropad(desired_length)(vector)) to be used in the ngram_dist and ngram_vals features
            ngram_zeros: the number of zero-bytes within ngram_vals
        """
        ngrams_list = SupportingFunctions.get_ngrams_across_nodes(cfg, n) if across_nodes else \
                      SupportingFunctions.get_ngrams_zeropad(cfg, n)
        ngrams, ngram_counts = order_by_occurence_then_value(ngrams_list)
        ngram_vals = ngram_zeros = -1 # Default values when mnemonics are used
        if isinstance(ngrams[0], tuple): # Opcode bytes are used instead of mnemonics
            ngram_vals = [int(byte) for byte in SupportingFunctions.concatenate_first_bytes(ngrams, len_vals)]
            ngram_zeros = sum(1 for num in ngram_vals if num == 0)
        return zeropad(len_dist, default=0)(ngram_counts), ngram_vals, len(ngram_counts), ngram_zeros
        #TODO: explain rationale for each proposed feature

    @staticmethod    
    def get_ngram_dist_features_subgraphs(cfg, n, len_dist, len_vals, len_zeros, across_nodes=True):
        """Extracts features based on ngrams, taking all nodes in all subgraphs into account
        
        Args:
            cfg: a CFG object
            n: the number of bytes in the ngram
            len_dist: the length of the ngram distribution count feature vector
            len_vals: the length of the feature vector formed by taking the first len_vals bytes of the sorted (with order_by_occurence_then_value(vector)) byte strings in the ngrams_list
            len_zeros: the amount of first bytes to take into account of the sorted (with order_by_occurence_then_value(vector)) byte strings in the ngrams_list when counting the number of zero-bytes among them
        KwArgs:
            across_nodes: whether to use the across_nodes method or the zeropad method for ngram extraction
        Returns:
            ngram_dist: a list representing the distribution of extracted ngrams from the graph downwards connected to the (sub_)root_node, using the zeropad or across_nodes method. The list is sorted from most number of occurences to least and contains the number of occurences of the first len_dist ngrams
            ngram_vals: a list composed of the first len_vals most occuring ngrams, concatenated
            ngram_len: the original amount of different extracted ngrams, before fixing them to a specific length (with zeropad(desired_length)(vector)) to be used in the ngram_dist and ngram_vals features
            ngram_zeros: the number of zero-bytes within ngram_vals
        """
        SupportingFunctions.set_subgraphs(cfg)
        if len(cfg.subgraphs) == 0: return [-1] * len_dist, [-1] * len_vals, -1, -1
        ngrams_list = []
        if across_nodes:
            for subgraph in cfg.subgraphs:
                ngrams_list.extend(SupportingFunctions.get_ngrams_across_nodes(cfg, n, graph=subgraph))
        else:
            for subgraph in cfg.subgraphs:
                ngrams_list.extend(SupportingFunctions.get_ngrams_zeropad(cfg, n, graph=subgraph))
        ngrams, ngram_counts = order_by_occurence_then_value(ngrams_list)
        ngram_vals = ngram_zeros = -1 # Default values when mnemonics are used
        if type(ngrams[0]) != type(tuple()): # Opcode bytes are used instead of mnemonics
            ngram_vals = [int(byte) for byte in SupportingFunctions.concatenate_first_bytes(ngrams, len_vals)]
            ngram_zeros = sum(1 for num in ngram_vals if num == 0)
        return zeropad(len_dist, default=0)(ngram_counts), ngram_vals, len(ngram_counts), ngram_zeros
        #FIXME: in YAML definition expression, use 'concatn' for the same effect as 'concatenate_first_bytes'
        #TODO: explain rationale for each proposed feature

    @staticmethod
    def get_nodesize_dist_features(cfg, len_dist, len_vals):
        """Extracts features based on node size
        
        Args:
            cfg: a CFG object
            len_dist: the length of the node size distribution count feature vector
            len_vals: the length of the feature vector formed by taking the first len_vals bytes of the sorted (with order_by_occurence_then_value(vector)) node sizes in the ngrams_list extraction
        Returns:
            nodesize_dist: a list representing the occurence count of the len_dist most occuring node sizes
            nodesize_vals: a list composed of the first len_vals most occuring node sizes
            nodesize_len: the original amount of different extracted node sizes, before fixing them to a specific length (with zeropad(desired_length)(vector)) to be used in the nodesize_dist and nodesize_vals features
        """
        node_sizes = []
        for node in cfg.model.nodes():
            if not node.size: node.size = 0
            node_sizes.append(node.size)
        sizes, size_counts = order_by_occurence_then_value(node_sizes)
        return zeropad(len_dist, default=0)(size_counts), zeropad(len_vals, default=0)(sizes), len(size_counts)

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


