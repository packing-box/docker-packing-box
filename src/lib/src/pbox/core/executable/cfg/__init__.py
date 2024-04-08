# -*- coding: UTF-8 -*-
from tinyscript import code, functools, logging, re, time
from tinyscript.helpers import zeropad, Capture, PathBasedDict, Timeout, TimeoutError

from ....helpers.mixins import *


_DEFAULT_EXCLUDE = set()


def __init_angr():
    angr_loggers = ["angr.*", "cle\..*", "pyvex.*"]
    configure_logging(reset=True, exceptions=angr_loggers)
    from ....helpers.config import _LOG_CONFIG
    for l in logging.root.manager.loggerDict:
        if any(re.match(al, l) for al in angr_loggers):
            logging.getLogger(l).setLevel([logging.WARNING, logging.DEBUG][_LOG_CONFIG[0]])
    from cle.backends.pe.regions import PESection
    try:
        code.insert_line(PESection.__init__, "from tinyscript.helpers import ensure_str", 0)
        code.replace(PESection.__init__, "pe_section.Name.decode()", "ensure_str(pe_section.Name)")
    except:
        pass
    for module in ["graph", "node"]:
        __import__(f"pbox.core.executable.cfg.{module}", globals(), locals(), [], 0)
lazy_load_module("angr", postload=__init_angr)


class CFG(GetItemMixin, ResetCachedPropertiesMixin):
    logger = logging.getLogger("executable.cfg")
    
    def __init__(self, target, engine=None, **kw):
        self.__target, self.engine = str(target), engine
    
    def compute(self, algorithm=None, timeout=None, **kw):
        l = self.__class__.logger
        if self.__project is None:
            l.error(f"{self.__target}: CFG project not created ; please set the engine")
            return
        from time import perf_counter
        t = perf_counter()
        try:
            with Capture() as c, Timeout(timeout or config['extract_timeout'], stop=True) as to:
                getattr(self.__project.analyses, f"CFG{algorithm or config['extract_algorithm']}") \
                    (fail_fast=False, resolve_indirect_jumps=True, normalize=True, model=self.model)
        except TimeoutError:
            l.warning(f"{self.__target}: Timeout reached when extracting CFG")
        except Exception as e:
            l.error(f"{self.__target}: Failed to extract CFG")
            l.exception(e)
        finally:
            self.__time = perf_counter() - t
            self._reset()
            for node in self.model.graph.nodes():
                if node.size:
                    node.byte_string = (tuple(insn.mnemonic for insn in node.block.disassembly.insns) \
                                        if config['opcode_mnemonics'] else bytes(insn.bytes[0] \
                                        for insn in node.block.disassembly.insns)) if config['only_opcodes'] \
                                       else node.block.bytes
            self.model.graph.root_node = self.root_node = self.model.get_any_node(self.model.project.entry) or \
                                                          next(_ for _ in self.model.nodes())
            self.model.graph._acyclic = False
    
    def iternodes(self, root_node=None, exclude=_DEFAULT_EXCLUDE):
        """ Iterate over nodes downwards from the provided root_node. """
        queue, visited = [root_node or self.graph.root_node], set()
        while queue:
            node = queue.pop(0)
            yield node
            visited.add(node.signature)
            for successor in node.successors:
                s = successor.signature
                if s in exclude or s in visited:
                    continue
                queue.append(successor)
    
    def iterinsns(self, root_node=None, exclude=_DEFAULT_EXCLUDE):
        """ Iterate over nodes downwards from the provided root_node, yielding pairs of opcode mnemonic and operand
             string. """
        for node in self.iternodes(root_node, exclude):
            if node.byte_string and not all(b == 0 for b in node.byte_string) and node.block:
                for i in node.block.disassembly.insns:
                    yield i.mnemonic, i.op_str
    
    @property
    def edges(self):
        return self.graph.edges
    
    @property
    def engine(self):
        return self.__engine
    
    @engine.setter
    def engine(self, name=None):
        name = name or config['angr_engine']
        cls = "UberEngine" if name in ["default", "vex"] else f"UberEngine{name.capitalize()}"
        try:
            self.__engine = getattr(angr.engines, cls)
            self.__project = angr.Project(self.__target, load_options={'auto_load_libs': False}, engine=self.__engine)
            self.model = self.__project.kb.cfgs.new_model(f"{self.__target}")
        except Exception as e:
            self.__engine = self.__project = None
            self.logger.exception(e)
    
    @property
    def graph(self):
        """ Compute and return the CFG. """
        try:
            next(_ for _ in self.model.graph.nodes())
        except StopIteration:
            self.compute()
        return self.model.graph
    
    @property
    def nodes(self):
        return self.graph.nodes
    
    @property
    def number_edges(self):
        try:
            return sum(1 for _ in self.edges())
        except AttributeError:
            return
    
    @property
    def number_nodes(self):
        try:
            return sum(1 for _ in self.nodes())
        except AttributeError:
            return
    
    @property
    def overlapping_nodes(self):
        """ Get all overlapping node pairs. """
        n, r = sorted(self.model.nodes(), key=lambda x: x.addr), set()
        for i in range(1, len(n)):
            if n[i-1].addr != n[i].addr and n[i-1].size and n[i-1].addr + n[i-1].size > n[i].addr:
                r.update((n[i-1], n[i]))
        return r
    
    # ----------------------------------------------- LAZY COMPUTATIONS ------------------------------------------------
    @cached_property
    def acyclic_graph(self):
        """ Compute and return the acyclic version of the CFG. """
        agraph, visited, stack = self.graph.__class__(), set(), [(None, self.graph.root_node)]
        while stack:
            predecessor, node = stack.pop()
            if node in visited:
                if config['store_loop_cut_info']:
                    if predecessor.irsb and predecessor.irsb[1]:
                        predecessor.irsb[1].append(node.byte_string)
                    elif predecessor.irsb:
                        predecessor.irsb[1] = [node.byte_string]
                    else:
                        predecessor.irsb = [0, [node.byte_string]]
                    if node.irsb:
                        node.irsb[0] += 1
                    else:
                        node.irsb = [1, []]
            else:
                visited.add(node)
                agraph.add_node(node)
                for successor in node.successors:
                    if successor not in visited:
                        agraph.add_edge(node, successor)
                    stack.append((node, successor))
        agraph._acyclic = False
        agraph.root_node = self.graph.root_node
        return agraph
    
    @cached_property
    def imported_apis(self):
        return list(self.model.project.loader.main_object.imports.keys())
    
    @cached_property
    def subgraphs(self):
        """ Get the list of subgraphs from the CFG. """
        def _graph2subgraph(graph, nodes):
            """ Create a subgraph from the input graph and list of nodes. """
            subgraph = graph.__class__()
            subgraph.add_nodes_from((n, graph.nodes[n]) for n in graph.subgraph(nodes))
            subgraph.add_edges_from((n, nbr, d) for n, nbrs in graph.adjacency() if n in nodes \
                                                for nbr, d in nbrs.items() if nbr in nodes)
            subgraph.root_node = nodes[0]
            return subgraph
        # start collecting subgraphs
        exclude = {node.signature for node in self.iternodes()} if config['exclude_duplicate_sigs'] else set()
        subgraphs, root_subgraph = [], list(self.iternodes())
        subgraphs.append(_graph2subgraph(self.acyclic_graph, root_subgraph))
        remaining_nodes = self.acyclic_graph.filter_nodes(self.nodes, root_subgraph)
        while remaining_nodes:
            for best_next_node in remaining_nodes:
                if not best_next_node.predecessors:
                    break
            sub_root_node = self.acyclic_graph.find_root(best_next_node, exclude)
            if sub_root_node.name == "PathTerminator":
                remaining_nodes = self.acyclic_graph.filter_nodes(remaining_nodes, [sub_root_node])
                continue
            subgraph = list(self.iternodes(sub_root_node, exclude))
            subgraphs.append(_graph2subgraph(self.acyclic_graph, subgraph))
            if config['exclude_duplicate_sigs']:
                exclude.update(node_signature(node) for node in subgraph)
            remaining_nodes = self.acyclic_graph.filter_nodes(remaining_nodes, subgraph)
        return subgraphs







class SupportingFunctions:

    @staticmethod    
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
        ngrams, ngram_counts = SupportingFunctions.order_by_occurence_then_value(ngrams_list)
        ngram_vals = ngram_zeros = -1 # Default values when mnemonics are used
        if type(ngrams[0]) != type(tuple()): # Opcode bytes are used instead of mnemonics
            ngram_vals = [int(byte) for byte in SupportingFunctions.concatenate_first_bytes(ngrams, len_vals)]
            ngram_zeros = sum(1 for num in ngram_vals if num == 0)
        return zeropad(len_dist, default=0)(ngram_counts), ngram_vals, len(ngram_counts), ngram_zeros
        #FIXME: in YAML definition expression, use 'concatn' for the same effect as 'concatenate_first_bytes'

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
        ngrams, ngram_counts = SupportingFunctions.order_by_occurence_then_value(ngrams_list)
        ngram_vals = ngram_zeros = -1 # Default values when mnemonics are used
        if type(ngrams[0]) != type(tuple()): # Opcode bytes are used instead of mnemonics
            ngram_vals = [int(byte) for byte in SupportingFunctions.concatenate_first_bytes(ngrams, len_vals)]
            ngram_zeros = sum(1 for num in ngram_vals if num == 0)
        return zeropad(len_dist, default=0)(ngram_counts), ngram_vals, len(ngram_counts), ngram_zeros
        #FIXME: in YAML definition expression, use 'concatn' for the same effect as 'concatenate_first_bytes'

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
        sizes, size_counts = SupportingFunctions.order_by_occurence_then_value(node_sizes)
        return zeropad(len_dist, default=0)(size_counts), zeropad(len_vals, default=0)(sizes), len(size_counts)

    @staticmethod
    def get_cfg_structure_features(cfg, include_cut_edges=True):
        """Extracts features based on cfg structure. If make_acyclic was called with store_loop_cut_info=True and this function with include_cut_edges=True, the information about the original graph before edges got cut by make_acyclic will be included in these cfg structure based features
          
        Args:
            cfg: a CFG object
        KwArgs:
            include_cut_edges: whether to include the stored information about edges that got cut
        Returns:
            avg_degrees_out: the average number of successors of all nodes in the subgraph downwards connected to the (sub_)root_node
            avg_degrees_in: the average number of predecessors of all nodes in the subgraph downwards connected to the (sub_)root_node
        """
        SupportingFunctions.make_acyclic(cfg)
        all_degrees_out = []
        all_degrees_in = []
        queue = [cfg.model.graph.name['root_node']]
        visited = set()
        while queue:
            node = queue.pop(0)
            visited.add(node)
            num_successors, num_predecessors = SupportingFunctions.get_num_neighbors(node, include_cut_edges=include_cut_edges)
            all_degrees_out.append(num_successors)
            all_degrees_in.append(num_predecessors)
            for successor in node.successors:
                if successor not in visited:
                    queue.append(successor)
        avg_degrees_out = sum(all_degrees_out) / len(all_degrees_out)
        avg_degrees_in = sum(all_degrees_in) / len(all_degrees_in)
        return avg_degrees_out, avg_degrees_in

    @staticmethod    
    def get_cfg_structure_features_subgraphs(cfg, include_cut_edges=True):
        """Extracts features based on cfg structure. If make_acyclic was called with store_loop_cut_info=True and this function with include_cut_edges=True, the information about the original graph before edges got cut by make_acyclic will be included in these cfg structure based features
        
        Args:
            cfg: a CFG object
        KwArgs:
            include_cut_edges: whether to include the stored information about edges that got cut
        Returns:
            avg_degrees_out: the average number of successors of all nodes in the subgraph downwards connected to the (sub_)root_node
            avg_degrees_in: the average number of predecessors of all nodes in the subgraph downwards connected to the (sub_)root_node
        """
        SupportingFunctions.set_subgraphs(cfg)
        if len(cfg.subgraphs) == 0:
            return (-1, ) * 2
        all_degrees_out = []
        all_degrees_in = []
        for subgraph in cfg.subgraphs:
            SupportingFunctions.make_acyclic(cfg, graph=subgraph)
            queue = [subgraph.name['root_node']]
            visited = set()
            while queue:
                node = queue.pop(0)
                visited.add(node)
                num_successors, num_predecessors = SupportingFunctions.get_num_neighbors(node, include_cut_edges=include_cut_edges, graph=subgraph)
                all_degrees_out.append(num_successors)
                all_degrees_in.append(num_predecessors)
                for successor in subgraph.successors(node):
                    if successor not in visited:
                        queue.append(successor)
        avg_degrees_out = sum(all_degrees_out) / len(all_degrees_out)
        avg_degrees_in = sum(all_degrees_in) / len(all_degrees_in)
        return avg_degrees_out, avg_degrees_in
    
    @staticmethod    
    def get_cfg_structure_features_all_nodes(cfg):
        """Extracts features based on cfg structure, taking all extracted nodes into account.
        
        Args:
            cfg: a CFG object
        Returns:
            avg_degrees_out: the average number of successors of all extracted nodes
            avg_degrees_in: the average number of predecessors of all extracted nodes
        """
        all_degrees_out = []
        all_degrees_in = []
        for node in cfg.model.nodes():
            num_successors, num_predecessors = SupportingFunctions.get_num_neighbors(node, include_cut_edges=True)
            all_degrees_out.append(num_successors)
            all_degrees_in.append(num_predecessors)
        avg_degrees_out = sum(all_degrees_out) / len(all_degrees_out)
        avg_degrees_in = sum(all_degrees_in) / len(all_degrees_in)
        return avg_degrees_out, avg_degrees_in

    @staticmethod    
    def get_subgraph_features(cfg):
        """Extracts features based on extracted subgraphs

        Args:
            cfg: a CFG object
        Returns:
            sg_max_size_is_root: boolean whether the root subgraph (containing the entry point) is the largest subgraph
            sg_max_size_ratio: ratio of (maximum size of extracted subgraphs) / (number of nodes extracted)
            sg_avg_size_ratio: ratio of (average size of extracted subgraphs) / (number of nodes extracted)
            sg_var_size_ratio: ratio of (variance of the size of extracted subgraphs) / (number of nodes extracted)
            sg_num_ratio: ratio of (amount of extracted subgraphs) / (number of nodes extracted)
            sg_perc_nopre: percentage of extracted subgraphs where the (sub_)root_node has no predecessor
        """
        SupportingFunctions.set_subgraphs(cfg)
        if len(cfg.subgraphs) == 0:
            return (-1, ) * 6
        num_nodes_found = len(cfg.model.nodes())
        sg_sizes = [len(sg) for sg in cfg.subgraphs]
        sg_max_size = max(sg_sizes)
        sg_max_size_is_root = sg_max_size == sg_sizes[0]
        sg_avg_size = sum(sg_sizes) / len(sg_sizes)
        sg_var_size = sum((i - sg_avg_size) ** 2 for i in sg_sizes) / len(sg_sizes)
        sg_num = len(cfg.subgraphs)
        sg_perc_nopre = sum([not sg.name['root_node'].predecessors for sg in cfg.subgraphs]) / sg_num
        return sg_max_size_is_root, sg_max_size / num_nodes_found, sg_avg_size / num_nodes_found, sg_var_size / num_nodes_found, sg_num / num_nodes_found, sg_perc_nopre

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


