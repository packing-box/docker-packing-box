# -*- coding: UTF-8 -*-
from tinyscript import code, logging, re
from tinyscript.helpers import Capture, Timeout, TimeoutError

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

