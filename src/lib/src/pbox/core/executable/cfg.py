# -*- coding: UTF-8 -*-
from tinyscript import code, logging, re, time
from tinyscript.helpers import Capture, Timeout, TimeoutError


def _config_angr_loggers():
    angr_loggers = ["angr.*", "cle\..*", "pyvex.*"]
    configure_logging(reset=True, exceptions=angr_loggers)
    from ...helpers.config import _LOG_CONFIG
    for l in logging.root.manager.loggerDict:
        if any(re.match(al, l) for al in angr_loggers):
            logging.getLogger(l).setLevel([logging.WARNING, logging.DEBUG][_LOG_CONFIG[0]])
    from cle.backends.pe.regions import PESection
    code.insert_line(PESection.__init__, "from tinyscript.helpers import ensure_str", 0)
    code.replace(PESection.__init__, "pe_section.Name.decode()", "ensure_str(pe_section.Name)")
lazy_load_module("angr", postload=_config_angr_loggers)


class CFG:
    logger = logging.getLogger("executable.cfg")
    
    def __init__(self, target, engine=None, **kw):
        self.__target, self.engine, self.byte_strings_set, self.subgraphs = str(target), engine, False, None
    
    def compute(self, algorithm=None, timeout=None, **kw):
        l = self.__class__.logger
        if self.__project is None:
            l.error(f"{self.__target}: CFG project not created ; please set the engine")
            return
        try:
            with Capture() as c, Timeout(timeout or config['extract_timeout'], stop=True) as to:
                getattr(self.__project.analyses, f"CFG{algorithm or config['extract_algorithm']}") \
                    (fail_fast=False, resolve_indirect_jumps=True, normalize=True, model=self.model)
        except TimeoutError:
            l.warning(f"{self.__target}: Timeout reached when extracting CFG")
        except Exception as e:
            l.error(f"{self.__target}: Failed to extract CFG")
            l.exception(e)
    
    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, name=None):
        name = config['angr_engine'] if name is None else name
        cls = "UberEngine" if name in ["default", "vex"] else f"UberEngine{name.capitalize()}"
        try:
            self._engine = getattr(angr.engines, cls)
            self.__project = angr.Project(self.__target, load_options={'auto_load_libs': False}, engine=self._engine)
            self.model = self.__project.kb.cfgs.new_model(f"{self.__target}")
        except Exception as e:
            self._engine = self.__project = None
            self.__class__.logger.exception(e)


class SupportingFunctions:

    @staticmethod
    def set_graph_attr(graph, attr, val=True):
        """Set attribute in the graph name to keep track of which operations have already been performed on the graph
        
        Args:
            graph: a NetworkX DiGraph
            attr: a string representing the attribute to set
        KwArgs:
            val: the value to set the attr to
        """
        if not graph.name: graph.name = {key: False for key in ['acyclic', 'depths_set', 'root_node']}
        graph.name[attr] = val

    @staticmethod
    def ensure_all_byte_strings(cfg, only_opcodes=False, mnemonics=False):
        """For each node in nodes, adds byte string of node instructions if not yet present
        
        Args:
            cfg: a CFG object
            only_opcodes: determines whether to store only the opcodes of each instruction in the node or also the operands
            mnemonic: if only storing opcodes, the mnemonic boolean determines whether to store the opcode bytes or the opcode mnemonics
        """
        if cfg.byte_strings_set: return
        for node in cfg.model.nodes():
            if node.size and not node.byte_string:
                try:
                    if only_opcodes: # To store only opcodes
                        if mnemonics: # To store opcode mnemonics
                            node.byte_string = tuple(insn.mnemonic for insn in node.block.disassembly.insns)
                        else: # To store opcode bytes
                            node.byte_string = bytes([insn.bytes[0] for insn in node.block.disassembly.insns])
                    else: # To store complete instruction (opcode+operand)
                        node.byte_string = node.block.bytes
                except KeyError:
                    pass
        cfg.byte_strings_set = True
    
    @staticmethod    
    def ensure_root_node(cfg):
        """Ensures the root node is extracted. If no node was found at the entry point, the first extracted node is taken as the root node. Use CFGEmulated as the extraction algorithm to ensure the node at the entry point is extracted. A feature that requires the root node to be extracted will first call this function.
        
        Args:
            cfg: a CFG object
        """        
        if cfg.model.graph.name and cfg.model.graph.name['root_node']: return
        root_node = cfg.model.get_any_node(cfg.model.project.entry)
        if not root_node: root_node = list(cfg.model.nodes())[0] # l.warning("No node extracted at the entry point. Taking first extracted node as root node.")
        SupportingFunctions.set_graph_attr(cfg.model.graph, 'root_node', val=root_node)
        
    @staticmethod
    def get_signature(node):
        """Get a signature that can be used to characterize the node
        
        Args:
            node: a CFG node extracted by Angr
        Returns:
            signature: a tuple of tuples of the addresses of its predecessors, itself, and its successors
        """
        return tuple(tuple(n.addr for n in l) for l in (node.predecessors, [node], node.successors))

    @staticmethod
    def make_acyclic(cfg, graph=None, root_node=None, store_loop_cut_info=True):
        """Makes the subgraph downwards connected to the (sub_)root_node acyclic
        
        Args:
            cfg: a CFG object
        KwArgs:
            graph: a NetworkX DiGraph containing the nodes for which the ngrams should be extracted, along with the edges between them
            root_node: a CFG node extracted by Angr. The node from which to start the process of making its successor subgraph acyclic
            store_loop_cut_info: determines whether information about edges that get cut will be stored in the parent node of the cut edge, for later use in features
            subgraphs: whether you're making all subgraphs acyclic, or just the root subgraph
        """
        SupportingFunctions.ensure_root_node(cfg)
        subgraphs = graph is not None
        if not graph: graph = cfg.model.graph
        if not root_node:
            root_node = graph.name['root_node']
            if not root_node: raise ValueError("No root_node set for the provided graph. Unable to proceed.")
        if graph.name and graph.name['acyclic']: return
        SupportingFunctions.ensure_all_byte_strings(cfg)
        seen = set()
        frontier = [(None, root_node)]
        while frontier:
            predecessor, node = frontier.pop()
            if node in seen:
                if store_loop_cut_info:
                    if predecessor.irsb and predecessor.irsb[1]: predecessor.irsb[1].append(node.byte_string)
                    elif predecessor.irsb:
                        predecessor.irsb[1] = [node.byte_string]
                    else:
                        predecessor.irsb = [0, [node.byte_string]]
                    if node.irsb:
                        node.irsb[0] += 1
                    else:
                        node.irsb = [1, []]
                graph.remove_edge(predecessor, node)
            else:
                seen.add(node)
                if subgraphs:
                    successors = graph.successors(node)
                else:
                    successors = node.successors
                for successor in successors:
                    frontier.append((node, successor))
        SupportingFunctions.set_graph_attr(graph, 'acyclic')

    @staticmethod                
    def make_acyclic_subgraphs(cfg, store_loop_cut_info=True):
        """Makes each subgraph in subgraphs acyclic
        
        Args:
            cfg: a CFG object
        KwArgs:
            store_loop_cut_info: determines whether information about edges that get cut will be stored in the parent node of the cut edge, for later use in features
        """
        SupportingFunctions.set_subgraphs(cfg)
        for subgraph in cfg.subgraphs:
            if subgraph.name and subgraph.name['acyclic']: return
            SupportingFunctions.make_acyclic(cfg, graph=subgraph, store_loop_cut_info=store_loop_cut_info)

    @staticmethod            
    def get_all_nodes(root_node, exclude=set()):
        """Get all nodes downwards connected to the provided root_node
        
        Args:
            root_node: a CFG node extracted by Angr
        KwArgs:
            exclude: a set of node signatures (calculated with get_signature(node)) to exclude in the extracted subgraph
        Returns:
            nodes: a list of nodes corresponding to a subgraph downwards connected to the provided root_node
        """
        nodes = []
        queue = [root_node]
        visited = set()
        noneSizeStreak = set()
        while queue:
            node = queue.pop(0)
            if node.size:
                noneSizeStreak.clear()
            else:
                noneSizeStreak.add(node)
            nodes.append(node)
            visited.add(SupportingFunctions.get_signature(node))
            suc_addrs = []
            sucsuc_addrs = []
            for successor in node.successors:
                signature = SupportingFunctions.get_signature(successor)
                if signature in exclude: continue
                tmp_sucsuc_addrs = []
                for sucsuc in successor.successors:
                    tmp_sucsuc_addrs.append(sucsuc.addr)
                if signature not in visited:
                    suc_addrs.append(successor.addr)
                    sucsuc_addrs.extend(tmp_sucsuc_addrs)
                    queue.append(successor)
        return nodes

    @staticmethod
    def get_A_not_B(A, B):
        """Get all nodes in A that have a different signature (calculated with get_signature(node)) than the nodes in B
        
        Args:
            A: a list of CFG nodes extracted by Angr
            B: a list of CFG nodes extracted by Angr
        Returns:
            A_not_B: a list of CFG nodes extracted by Angr with the nodes from A that have a different signature (calculated with get_signature(node)) than the nodes in B
        """
        A_not_B = []
        B_sigs = [SupportingFunctions.get_signature(node) for node in B]
        for node in A:
            if SupportingFunctions.get_signature(node) not in B_sigs:
                A_not_B.append(node)
        return A_not_B  

    @staticmethod
    def remove_duplicates(nodes):
        """Get all nodes with a unique signature (calculated with get_signature(node))
        
        Args:
            nodes: a list of CFG nodes extracted by Angr
        Returns:
            out: a list of CFG nodes extracted by Angr, with a unique signature (calculated with get_signature(node))
        """
        out = []
        visited = set()
        for node in nodes:
            signature = SupportingFunctions.get_signature(node)
            if signature not in visited:
                visited.add(signature)
                out.append(node)
        return out

    @staticmethod
    def valid_sub_root_node(sub_root_node, already_checked_nodes):
        """Verifies if using this sub_root_node to extract a subgraph will yield a subgraph that contains the original node from which this sub_root_node was calculated (with get_root_node(node))
        
        Args:
            sub_root_node: a CFG node extracted by Angr
            already_checked_nodes: a list of CFG nodes extracted by Angr containing the nodes that have already been visited when getting the downward subgraph connected to a successor of sub_root_node
        Returns:
            valid: boolean, whether this sub_root_node can be taken or not to ensure its subgraph will contain the original node from which this sub_root_node was calculated (with get_root_node(node))
            already_checked_nodes: a list of CFG nodes extracted by Angr containing the nodes that have already been visited when getting the downward subgraph connected to a successor of sub_root_node
        """
        sub_root_successor = SupportingFunctions.remove_duplicates(sub_root_node.successors)
        if len(sub_root_successor) > 1:
            sub_root_successor = SupportingFunctions.get_A_not_B(sub_root_successor, [sub_root_node])
        if len(sub_root_successor) > 1:
            sub_root_successor = SupportingFunctions.get_A_not_B(sub_root_successor, already_checked_nodes)
        if len(sub_root_successor) > 1:
            sub_root_successor = SupportingFunctions.get_A_not_B(sub_root_successor, [n for n in sub_root_successor if n.addr + n.size != sub_root_node.addr])
        if len(sub_root_successor) == 0: return False, []
        assert len(sub_root_successor) == 1
        subgraph = SupportingFunctions.get_all_nodes(sub_root_successor[0])
        new_nodes = SupportingFunctions.get_A_not_B(subgraph, already_checked_nodes)
        if len(subgraph) == len(new_nodes):
            already_checked_nodes.extend([sub_root_node])
            already_checked_nodes.extend(new_nodes)
            return True, already_checked_nodes
        else:
            return False, []
     
    @staticmethod   
    def get_root_node(node, exclude=set()):
        """Get the best candidate for a (sub_)root_node by traversing the graph upward, ensuring the input node is still part of the subgraph extracted from the returned (sub_)root_node
        
        Args:
            node: a CFG node extracted by Angr
        KwArgs:
            exclude: a set of node signatures (calculated with get_signature(node)) to exclude in the extracted subgraph
        Returns:
            sub_root_node: the best candidate for a (sub_)root_node by traversing the graph upward, ensuring the input node is still part of the subgraph extracted from the returned (sub_)root_node
        """
        visited = set()
        sub_root_node = node
        already_checked_nodes = [node]
        while sub_root_node.predecessors:
            first_predecessor = sub_root_node.predecessors[0]
            if SupportingFunctions.get_signature(sub_root_node) == SupportingFunctions.get_signature(first_predecessor): break
            signature = SupportingFunctions.get_signature(first_predecessor)
            if signature in visited or signature in exclude: break
            valid, already_checked_nodes = SupportingFunctions.valid_sub_root_node(first_predecessor, already_checked_nodes)
            if not valid: break
            sub_root_node = first_predecessor
            visited.add(SupportingFunctions.get_signature(sub_root_node))
        return sub_root_node

    @staticmethod    
    def list_to_subgraph(cfg_model, subgraph):
        """Convert a list with nodes forming a subgraph to a NetworkX DiGraph object containing those nodes and the edges between them, according to the edges in the original cfg_model.graph
        
        Args:
            cfg_model: a CFG Model used by Angr to store the extracted nodes
        Returns:
            SG: a NetworkX DiGraph, containing the nodes in the subgraph list and the edges between them, according to the edges in the original cfg_model.graph
        """
        SG = cfg_model.graph.__class__()
        G = cfg_model.graph
        SG.add_nodes_from((n, G.nodes[n]) for n in G.subgraph(subgraph))
        SG.add_edges_from((n, nbr, d) for n, nbrs in G.adjacency() if n in subgraph for nbr, d in nbrs.items() if nbr in subgraph)
        SupportingFunctions.set_graph_attr(SG, 'root_node', val=subgraph[0])
        return SG

    @staticmethod
    def set_subgraphs(cfg, exclude_dup_signatures=True):
        """Get all (intentionally or not intentionally) detached graphs so that all signatures of the input nodes (calculated with get_signature(node)) appear in at least one (and at most one, depending on exclude_dup_signatures) of the returned graphs
        
        Args:
            cfg: a CFG object
        KwArgs:
            exclude_dup_signatures: whether to avoid nodes in the returned graphs with the same signature (calculated with get_signature(node))
        """
        if cfg.subgraphs: return
        nodes = list(cfg.model.nodes())
        SupportingFunctions.make_acyclic(cfg)
        root_subgraph = SupportingFunctions.get_all_nodes(nodes[0])
        exclude = set()
        if exclude_dup_signatures: exclude.update(SupportingFunctions.get_signature(node) for node in root_subgraph)
        subgraphs = []
        subgraphs.append(SupportingFunctions.list_to_subgraph(cfg.model, root_subgraph))
        remaining_nodes = SupportingFunctions.get_A_not_B(nodes, root_subgraph)
        while remaining_nodes:
            for best_next_node in remaining_nodes:
                if not best_next_node.predecessors: break
            sub_root_node = SupportingFunctions.get_root_node(best_next_node, exclude)
            if sub_root_node.name == "PathTerminator":
                remaining_nodes = SupportingFunctions.get_A_not_B(remaining_nodes, [sub_root_node])
                continue
            SupportingFunctions.make_acyclic(cfg, root_node=sub_root_node)
            subgraph = SupportingFunctions.get_all_nodes(sub_root_node, exclude)
            subgraphs.append(SupportingFunctions.list_to_subgraph(cfg.model, subgraph))
            if exclude_dup_signatures: exclude.update(SupportingFunctions.get_signature(node) for node in subgraph)
            remaining_nodes = SupportingFunctions.get_A_not_B(remaining_nodes, subgraph)
        cfg.subgraphs = subgraphs

    @staticmethod
    def get_ngrams_zeropad(cfg, n, graph=None):
        """Gets a list of ngrams, using the zeropad method (zero-padding node.byte_string to a length of n if it is shorter)
        
        Args:
            cfg: a CFG object
            n: the number of bytes in the ngram
        KwArgs:
            graph: a NetworkX DiGraph containing the nodes for which the ngrams should be extracted, along with the edges between them
        Returns:
            ngrams_list: a list of ngrams extracted from the nodes in the graph downwards connected to the (sub_)root_node, using the zeropad method
        """
        SupportingFunctions.ensure_all_byte_strings(cfg)
        if graph: root_node = graph.name['root_node']
        else: root_node = cfg.model.graph.name['root_node']
        ngrams_list = []
        queue = [root_node]
        visited = set()
        mnemonics = type(root_node.byte_string) == type(tuple())
        while queue:
            node = queue.pop(0)
            if node.byte_string:
                if mnemonics:
                    byte_string = node.byte_string + ('',) * (n - len(node.byte_string))
                else:
                    byte_string = node.byte_string.ljust(n,b'\0')
                ngrams_list.extend([node.byte_string[i:i+n] for i in range(len(node.byte_string) - n + 1)])
            if node.irsb and node.irsb[1]:
                for cut_successor_byte_string in node.irsb[1]:
                    if cut_successor_byte_string:
                        if mnemonics:
                            byte_string = cut_successor_byte_string + ('',) * (n - len(cut_successor_byte_string))
                        else:
                            byte_string = cut_successor_byte_string.ljust(n,b'\0')
                        ngrams_list.extend([byte_string[i:i+n] for i in range(len(byte_string) - n + 1)])
            if graph:
                successors = graph.successors(node)
            else:
                successors = node.successors
            for successor in successors:
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
        return ngrams_list
        
    @staticmethod
    def ensure_acyclic(cfg, graph, root_node=None):
        """Ensures the (sub)graph is acyclic
        
        Args:
            cfg: a CFG object
            graph: a NetworkX DiGraph containing the nodes and edges between them
        KwArgs:
            root_node: a CFG node extracted by Angr, should be a (sub_)root_node when calling the function. If it is None, it will get set to the (sub_)root_node
        Returns:
            root_node: the provided (sub_)root_node, or the newly extracted one, if None was provided
        """
        if graph:
            SupportingFunctions.make_acyclic(cfg, graph=graph)
            if not root_node: root_node = graph.name['root_node']
        else:
            SupportingFunctions.make_acyclic(cfg)
            if not root_node: root_node = cfg.model.graph.name['root_node']
        if not root_node: raise ValueError("No root_node set for the provided graph. Unable to proceed.")
        return root_node

    @staticmethod
    def get_ngrams_across_nodes(cfg, n, node=None, pre_bytes=None, graph=None, recursion_limit=None):
        """Gets a list of ngrams, using the across_nodes method (includes the ngrams that consist of instuctions of connected nodes)
        
        Args:
            n: the number of bytes in the ngram
        KwArgs:
            node: a CFG node extracted by Angr, should be a (sub_)root_node when calling the function
            pre_bytes: memory of the byte_string preceding the current node
            graph: a NetworkX DiGraph containing the nodes for which the ngrams should be extracted, along with the edges between them
            recursion_limit: an integer equal to the max allowed amount of recursive function calls. If not set, it will be set to the system's recursionlimit
        Returns:
            ngrams_list: a list of ngrams extracted from the nodes in the graph downwards connected to the (sub_)root_node, using the across_nodes method
        """
        if not node: node = SupportingFunctions.ensure_acyclic(cfg, graph)
        if not recursion_limit:
            import sys
            recursion_limit = sys.getrecursionlimit()
        if recursion_limit <= 50: return []
        ngrams_list = []
        if node.byte_string:
            if pre_bytes is None:
                pre_bytes = type(node.byte_string)()
            byte_string = pre_bytes + node.byte_string
            ngrams_list = [byte_string[i:i+n] for i in range(len(byte_string) - n + 1)]
            pre_bytes = byte_string[-n+1:]
        if node.irsb and node.irsb[1]:
            for cut_successor_byte_string in node.irsb[1]:
                if cut_successor_byte_string:
                    if pre_bytes is None:
                        pre_bytes = type(cut_successor_byte_string)()
                    byte_string = pre_bytes + cut_successor_byte_string
                    ngrams_list.extend([byte_string[i:i+n] for i in range(len(byte_string) - n + 1)])
        if graph:
            successors = graph.successors(node)
        else:
            successors = node.successors
        for successor in successors:
            ngrams_list.extend(SupportingFunctions.get_ngrams_across_nodes(cfg, n, node=successor, pre_bytes=pre_bytes, graph=graph, recursion_limit=recursion_limit-1))
        return ngrams_list

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

    @staticmethod    
    def zero_pad_vector(vector, desired_length):
        """Ensures the returned vector has the desired_length, either by zeropadding, or cutting the input vector in size
        
        Args:
            vector: a list of arbitrary length
            desired_length: the desired length of the output list, in number of elements
        Returns:
            desired_length_vector: a list with the desired_length, either by zeropadding, or cutting the input vector in size
        """
        return vector[:desired_length] + [0] * max(0, desired_length - len(vector))

    @staticmethod
    def concatenate_first_bytes(ngrams, M):
        """Concatenate the first M bytes of the (sorted) ngrams list
        
        Args:
            ngrams: a list of ngrams
            M: the number of concatenated ngram bytes to return
        Returns:
            result: a byte string of the first M bytes of the (sorted) ngrams list
        """
        result = type(ngrams[0])()
        for ngram in ngrams:
            result += ngram[:M]
            if len(result) >= M:
                break
        return result[:M]

    @staticmethod
    def set_depth(cfg, node=None, graph=None, first_iter=True, recursion_limit=None):
        """Sets the depth parameter for each node in the subgraph downwards connected to the (sub_)root_node
        
        Args:
            cfg: a CFG object
        KwArgs:
            node: a CFG node extracted by Angr, should be a (sub_)root_node when calling the function
            graph: a NetworkX DiGraph containing the nodes for which the depths should be set, along with the edges between them
            first_iter: a boolean indicating if this is the first iteration of calling set_depth recursively
            recursion_limit: an integer equal to the max allowed amount of recursive function calls. If not set, it will be set to the system's recursionlimit 
        """
        if first_iter:
            if graph:
                if graph.name and graph.name['depths_set']: return
            else:
                if cfg.model.graph.name and cfg.model.graph.name['depths_set']: return
            node = SupportingFunctions.ensure_acyclic(cfg, graph, root_node=node)
            if not recursion_limit:
                import sys
                recursion_limit = sys.getrecursionlimit()
        if recursion_limit <= 50: return 0
        if not node.size: node.size = 0
        max_successor_depth = 0
        if graph:
            successors = graph.successors(node)
        else:
            successors = node.successors
        for successor in successors:
            depth = SupportingFunctions.set_depth(cfg, node=successor, graph=graph, first_iter=False, recursion_limit=recursion_limit-1)
            if depth > max_successor_depth: max_successor_depth = depth
        node.soot_block = {'depth': node.size + max_successor_depth, 'idx': None}
        if first_iter:
            if graph:
                SupportingFunctions.set_graph_attr(graph, 'depths_set')
            else:
                SupportingFunctions.set_graph_attr(cfg.model.graph, 'depths_set')
        return node.soot_block['depth']

    @staticmethod    
    def set_depth_subgraphs(cfg):
        """Sets the depth parameter for each node in the subgraph downwards connected to the (sub_)root_node of each subgraph in subgraphs
        
        Args:
            cfg: a CFG object
        """
        SupportingFunctions.set_subgraphs(cfg)
        for subgraph in cfg.subgraphs:
            SupportingFunctions.set_depth(cfg, graph=subgraph)

    @staticmethod
    def get_num_neighbors(node, include_cut_edges=True, graph=None):
        """Gets the original number of neighbors the node used to have before making the graph acyclic (if make_acyclic was called with store_loop_cut_info=True), otherwise (or if include_cut_edges=False) gets the new number of neighbors of the node after making the graph acyclic
        
        Args:
            node: a CFG node extracted by Angr
        KwArgs:
            include_cut_edges: whether to include the stored information about edges that got cut
            graph: a NetworkX DiGraph containing the nodes for which the neighbor relationships should be extracted, along with the edges between them
        Returns:
            num_successors: the number of successors of the node
            num_predecessors: the number of predecessors of the node
        """
        if graph:
            num_successors = len(list(graph.successors(node)))
            num_predecessors = len(list(graph.predecessors(node)))
        else:
            num_successors = len(node.successors)
            num_predecessors = len(node.predecessors)
        if include_cut_edges and node.irsb:
            if node.irsb[1]:
                num_successors += len(node.irsb[1])
            elif node.irsb[0]:
                num_predecessors += node.irsb[0]
        return num_successors, num_predecessors

    @staticmethod    
    def get_overlapping_nodes(nodes):
        """Gets all overlapping node pairs
         
        Args:
            nodes: a list of CFG nodes extracted by Angr
        Returns:
            overlapping_nodes: a list of overlapping node pairs
        """
        n = nodes.copy()
        n.sort(key=lambda x: x.addr)
        overlapping_nodes = []
        for i in range(1, len(n)):
            if n[i-1].addr != n[i].addr and n[i-1].size and n[i-1].addr + n[i-1].size > n[i].addr:
                overlapping_nodes.append((n[i-1], n[i]))
        return overlapping_nodes

    @staticmethod    
    def get_insn_tuples(cfg):
        """Gets all tuples of (opcode mnemonic, operand string) of all extracted nodes, avoiding duplicate counts by using the get_signature supporting function
          
        Args:
            cfg: a CFG object
        """
        insn_tuples = []
        visited = set()
        for node in cfg.model.nodes():
            signature = SupportingFunctions.get_signature(node)
            if signature in visited: continue
            visited.add(signature)
            if node.byte_string and not all(byte == 0 for byte in node.byte_string) and node.block: insn_tuples.extend([(i.mnemonic, i.op_str) for i in node.block.disassembly.insns])
        return insn_tuples


class FeatureExtractionFunctions:

    @staticmethod
    def found_node_at_entry_feature(cfg):
        """Whether a node was found at the entry point
        
        Args:
            cfg: a CFG object
        Returns:
            found_node_at_entry: whether a node was found at the entry point
        """
        return cfg.model.get_any_node(cfg.model.project.entry) is not None

    @staticmethod    
    def num_nodes(cfg):
        """Number of nodes found by Angr
        
        Args:
            cfg: a CFG object
        Returns:
            num_nodes_found: number of nodes found by Angr
        """
        return len(cfg.model.nodes())

    @staticmethod    
    def num_edges(cfg):
        """Number of edges in the graph
        
        Args:
            cfg: a CFG object
        Returns:
            num_edges_found: number of edges in the graph
        """
        if cfg.model.graph.name and cfg.model.graph.name['acyclic']: raise RuntimeError("For feature consistency, the number of edges must always be counted BEFORE making the graph acyclic.")
        return len(cfg.model.graph.edges())
        
    @staticmethod    
    def get_overlapping_nodes_ratio(cfg):
        """Gets all overlapping node pairs
         
        Args:
            cfg: a CFG object
        Returns:
            overlapping_nodes_ratio: a ratio of the number of overlapping nodes over the total number of nodes
        """
        nodes = list(cfg.model.nodes())
        nodes.sort(key=lambda x: x.addr)
        num_overlapping_nodes = 0
        for i in range(1, len(nodes)):
            if nodes[i-1].addr != nodes[i].addr and nodes[i-1].size and nodes[i-1].addr + nodes[i-1].size > nodes[i].addr:
                num_overlapping_nodes += 2
        return num_overlapping_nodes / len(nodes)

    @staticmethod    
    def get_sink_features(cfg):
        """Extracts features based on sink nodes
        
        Args:
            cfg: a CFG object
        Returns:
            frac_pts: fraction of PathTerminator nodes
            frac_nosuc: fraction of nodes with no successors
        """
        num_pathterminators = 0
        num_nosuccessors = 0
        for node in cfg.model.nodes():
            if node.name == "PathTerminator": num_pathterminators += 1
            if not node.successors: num_nosuccessors += 1
        num_nodes = len(cfg.model.nodes())
        frac_pts = num_pathterminators / num_nodes
        frac_nosuc = num_nosuccessors / num_nodes
        return frac_pts, frac_nosuc

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
            ngram_len: the original amount of different extracted ngrams, before fixing them to a specific length (with zero_pad_vector(vector, desired_length)) to be used in the ngram_dist and ngram_vals features
            ngram_zeros: the number of zero-bytes within ngram_vals
        """
        if across_nodes:
            ngrams_list = SupportingFunctions.get_ngrams_across_nodes(cfg, n)
        else:
            ngrams_list = SupportingFunctions.get_ngrams_zeropad(cfg, n)
        ngrams, ngram_counts = SupportingFunctions.order_by_occurence_then_value(ngrams_list)
        ngram_vals = ngram_zeros = -1 # Default values when mnemonics are used
        if type(ngrams[0]) != type(tuple()): # Opcode bytes are used instead of mnemonics
            ngram_vals = [int(byte) for byte in SupportingFunctions.concatenate_first_bytes(ngrams, len_vals)]
            ngram_zeros = sum(1 for num in ngram_vals if num == 0)
        return SupportingFunctions.zero_pad_vector(ngram_counts, len_dist), ngram_vals, len(ngram_counts), ngram_zeros

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
            ngram_len: the original amount of different extracted ngrams, before fixing them to a specific length (with zero_pad_vector(vector, desired_length)) to be used in the ngram_dist and ngram_vals features
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
        return SupportingFunctions.zero_pad_vector(ngram_counts, len_dist), ngram_vals, len(ngram_counts), ngram_zeros

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
            nodesize_len: the original amount of different extracted node sizes, before fixing them to a specific length (with zero_pad_vector(vector, desired_length)) to be used in the nodesize_dist and nodesize_vals features
        """
        node_sizes = []
        for node in cfg.model.nodes():
            if not node.size: node.size = 0
            node_sizes.append(node.size)
        sizes, size_counts = SupportingFunctions.order_by_occurence_then_value(node_sizes)
        return SupportingFunctions.zero_pad_vector(size_counts, len_dist), SupportingFunctions.zero_pad_vector(sizes, len_vals), len(size_counts)

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
        if len(cfg.subgraphs) == 0: return (-1, ) * 2
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
    def both_signatures(cfg, len_approx, len_exact, include_cut_edges=True):
        """Extracts features (signatures) based on cfg structure
        
        Args:
            cfg: a CFG object
            len_approx: the length of the 'approximate' (cf. reference) feature vector (signature)
            len_exact: the length of the 'exact' (cf. reference) feature vector (signature)
        KwArgs:
            include_cut_edges: whether to include the stored information about edges that got cut
        Returns:
            approx_signature: a list representing the 'approximate' (cf. reference) feature vector (signature)
            exact_signature: a list representing the 'exact' (cf. reference) feature vector (signature)
        Reference: https://ieeexplore.ieee.org/document/8170793
        """
        SupportingFunctions.set_depth(cfg)
        approx_signature = []
        idx = 1
        root_node = cfg.model.graph.name['root_node']
        root_node.soot_block['idx'] = idx
        queue = [root_node]
        visited = set()
        while queue and len(approx_signature) < len_approx:
            current = queue.pop(0)
            visited.add(current)
            num_successors, num_predecessors = SupportingFunctions.get_num_neighbors(current, include_cut_edges=include_cut_edges)
            num_predecessors = min(num_predecessors, 63)
            feature = (num_successors << 6) | num_predecessors
            approx_signature.append(feature)
            successors = current.successors
            successors.sort(key=lambda x: -x.soot_block['depth'])
            for successor in successors:
                if successor not in visited:
                    idx += 1
                    successor.soot_block['idx'] = idx
                    queue.append(successor)
        exact_signature = []
        queue = []
        queue.append(root_node)
        while queue and len(exact_signature) < len_exact:
            current = queue.pop(0)
            visited.add(current)
            successors = current.successors
            successors.sort(key=lambda x: -x.soot_block['depth'])
            for successor in successors:
                exact_signature.append(successor.soot_block['idx'])
                queue.append(successor)
            for _ in range(2-len(successors)):
                exact_signature.append(0)
        return SupportingFunctions.zero_pad_vector(approx_signature, len_approx) , SupportingFunctions.zero_pad_vector(exact_signature, len_exact)

    # Function can be used if exact_signature turns out not to be that useful
    @staticmethod
    def approximate_signature(cfg, len_approx, include_cut_edges=True):
        """Extracts features (signature) based on cfg structure

        Args:
            cfg: a CFG object
            len_approx: the length of the 'approximate' (cf. reference) feature vector (signature)
        KwArgs:
            include_cut_edges: whether to include the stored information about edges that got cut
        Returns:
            approx_signature: a list representing the 'approximate' (cf. reference) feature vector (signature)
        Reference: https://ieeexplore.ieee.org/document/8170793
        """
        SupportingFunctions.set_depth(cfg)
        approx_signature = []
        queue = [cfg.model.graph.name['root_node']]
        visited = set()
        while queue:
            current = queue.pop(0)
            visited.add(current)
            num_successors, num_predecessors = SupportingFunctions.get_num_neighbors(current, include_cut_edges=include_cut_edges)
            num_predecessors = min(num_predecessors, 63)
            feature = (num_successors << 6) | num_predecessors
            approx_signature.append(feature)
            successors = current.successors
            successors.sort(key=lambda x: -x.soot_block['depth'])
            for successor in successors:
                if successor not in visited:
                    queue.append(successor)
        return SupportingFunctions.zero_pad_vector(approx_signature, len_approx)

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
        if len(cfg.subgraphs) == 0: return (-1, ) * 6
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
        insn_tuples = SupportingFunctions.get_insn_tuples(cfg)
        num_jump_insns = num_indirect_jump_insns = 0
        for insn_tuple in insn_tuples:
            if insn_tuple[0] in jump_mnemonics:
                num_jump_insns += 1
                if insn_tuple[1].startswith('0x'):
                    num_indirect_jump_insns += 1
        groups = {
            "Return value": {"rax", "eax", "ax", "ah", "al"},
            "General-Purpose Registers": {"rbx", "rcx", "rdx", "ebx", "ecx", "edx", "bx", "bh", "bl", "cx", "ch", "cl", "dx", "dh", "dl"},
            "Segment Registers": {"cs", "ds", "es", "fs", "gs", "ss"},
            "Function arguments": {"rsi", "rdi", "esi", "edi"},
            "Stack Registers": {"rbp", "rsp", "ebp", "esp"},
            "Instruction Pointer": {"rip", "eip"},
            "Flags Register": {"rflags", "eflags"},
            "Floating-Point Registers": set("xmm{}".format(i) for i in range(16))
        }
        register_count = {group: 0 for group in groups}
        trans_table = str.maketrans({',': ' ', '+': ' ', '*': ' ', '[': ' ', ']': ' ', '-': ' '})
        operand_tokens = sum((it[1].translate(trans_table).split() for it in insn_tuples), [])
        for token in operand_tokens:
            for group, register_set in groups.items():
                if token in register_set:
                    register_count[group] += 1
        register_type_counts = list(register_count.values())
        ratio_indir_jmps = -1 # Default value if num_jump_insns is zero
        if num_jump_insns: ratio_indir_jmps = num_indirect_jump_insns / num_jump_insns
        return ratio_indir_jmps, [-1 if sum(register_type_counts) == 0 else count / sum(register_type_counts) for count in register_type_counts], [-1 if len(insn_tuples) == 0 else count / len(insn_tuples) for count in register_type_counts]


