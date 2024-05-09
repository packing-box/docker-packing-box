# -*- coding: UTF-8 -*-
from tinyscript import code, logging, re, functools
from tinyscript.helpers import Capture, Timeout, TimeoutError

from ....helpers.mixins import *


__all__ = ["CFG"]

_DEFAULT_EXCLUDE = set()


def __init_angr():
    angr_loggers = ["angr.*", "cle\..*", "pyvex.*"]
    configure_logging(reset=True, exceptions=angr_loggers)
    from ....helpers.config import _LOG_CONFIG
    for l in logging.root.manager.loggerDict:
        if any(re.match(al, l) for al in angr_loggers):
            logging.getLogger(l).setLevel([logging.WARNING, logging.DEBUG][_LOG_CONFIG[0]])
    from angr.misc.picklable_lock import PicklableLock
    from threading import RLock
    PicklableLock._LOCK = RLock
    from cle.backends.pe.regions import PESection
    try:
        code.insert_line(PESection.__init__, 1, "from tinyscript.helpers import ensure_str")
        code.replace(PESection.__init__, "pe_section.Name.decode()", "ensure_str(pe_section.Name)")
    except:
        pass
    for module in ["graph", "node"]:
        __import__(f"pbox.core.executable.cfg.{module}", globals(), locals(), [], 0)
lazy_load_module("angr", postload=__init_angr)


class CFG(GetItemMixin, ResetCachedPropertiesMixin):
    engines = {k: getattr(angr.engines, "UberEngine" if k in ["default", "vex"] else f"UberEngine{k.capitalize()}") \
               for k in ANGR_ENGINES}
    logger = logging.getLogger("executable.cfg")
    
    def __init__(self, target, engine=None, **kw):
        self.__target = str(target)
        try:
            self.__project = angr.Project(self.__target, load_options={'auto_load_libs': False},
                                          engine=self.engines[engine or config['angr_engine']])
            self.model = self.__project.kb.cfgs.new_model(f"{self.__target}")
        except Exception as e:
            self.__project = self.model = None
            self.logger.exception(e)
    
    def compute(self, algorithm=None, timeout=None, **kw):
        l = self.__class__.logger
        if self.__project is None:
            l.error(f"{self.__target}: CFG project not created")
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
                    try:
                        node.byte_string = (tuple(insn.mnemonic for insn in node.block.disassembly.insns) \
                                            if config['opcode_mnemonics'] else bytes(insn.bytes[0] \
                                            for insn in node.block.disassembly.insns)) if config['only_opcodes'] \
                                           else node.block.bytes
                    except KeyError:
                        pass
            self.model.graph.root_node = self.root_node = self.model.get_any_node(self.model.project.entry) or \
                                                          next(_ for _ in self.model.nodes())
            self.model.graph._acyclic = False
    
    def iternodes(self, root_node=None, exclude=_DEFAULT_EXCLUDE):
        """ Iterate over nodes downwards from the provided root_node. """
        queue, visited = [root_node or self.graph.root_node], set()
        while queue:
            node = queue.pop(0)
            sig = node.signature
            if sig in exclude or sig in visited:
                continue
            yield node
            visited.add(sig)
            for successor in self.graph.successors(node):
                sig = successor.signature
                if sig in exclude or sig in visited:
                    continue
                queue.append(successor)
    
    def iterinsns(self, root_node=None, exclude=_DEFAULT_EXCLUDE):
        """ Iterate over nodes downwards from the provided root_node, yielding pairs of opcode mnemonic and operand
             string. """
        for node in self.iternodes(root_node, exclude):
            if node.byte_string and not all(b == 0 for b in node.byte_string) and node.block:
                for i in node.block.disassembly.insns:
                    yield i.mnemonic, i.op_str

    @staticmethod
    def sortedhist(vector):
        """Orders the elements of the input list first by occurence, then by value, yielding a sorted histogram
        
        :param vector: a list of unordered elements
        :return: (list of ordered elements; first by occurence; then by value,
                list of number of occurences of the elements in ordered list)
        """
        from collections import Counter
        return list(map(list, zip(*sorted(Counter(vector).items(), key=lambda x: (x[1], x[0]), reverse=True))))
    
    @staticmethod
    def to_acyclic(graph, root_node=None):
        agraph, visited, stack = graph.__class__(), set(), [(None, root_node or graph.root_node)]
        while stack:
            predecessor, node = stack.pop()
            if node in visited:
                agraph.remove_edge(predecessor, node)
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
                for successor in graph.successors(node):
                    agraph.add_edge(node, successor)
                    stack.append((node, successor))
        agraph._acyclic = True
        agraph.root_node = root_node or graph.root_node
        return agraph
    
    @property
    def edges(self):
        return self.graph.edges

    @property
    def entry(self):
        return self.model.project.entry
    
    @property
    def graph(self):
        """ Compute and return the CFG. """
        try:
            next(_ for _ in self.model.graph.nodes())
        except (AttributeError, StopIteration):
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
        return self.__class__.to_acyclic(self.graph)
    
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
        subgraphs.append(self.__class__.to_acyclic(_graph2subgraph(self.graph, root_subgraph)))
        remaining_nodes = self.graph.filter_nodes(self.nodes, root_subgraph)
        while remaining_nodes:
            for best_next_node in remaining_nodes:
                if not best_next_node.predecessors:
                    break
            if best_next_node.signature in exclude:
                remaining_nodes = self.graph.filter_nodes(remaining_nodes, [best_next_node])
                continue
            sub_root_node = self.graph.find_root(best_next_node, exclude)
            exclude.update(sub_root_node.signature)
            if sub_root_node.name == "PathTerminator":
                remaining_nodes = self.graph.filter_nodes(remaining_nodes, [sub_root_node])
                continue
            subgraph = list(self.iternodes(sub_root_node, exclude))
            subgraphs.append(self.__class__.to_acyclic(_graph2subgraph(self.graph, subgraph)))
            if config['exclude_duplicate_sigs']:
                exclude.update(node.signature for node in subgraph)
            remaining_nodes = self.graph.filter_nodes(remaining_nodes, subgraph)
        return subgraphs

    @cached_property
    def nodesize_hist(self):
        """ Gets a sorted histogram of node sizes. """
        return self.__class__.sortedhist([node.size if node.size else 0 for node in self.nodes])

    @functools.lru_cache
    def ngram_hist(self, n, across_nodes=True, all_subgraphs=False):
        """ Gets a sorted histogram of ngrams. """
        from hashlib import sha1
        return self.__class__.sortedhist([b''.join(map(lambda x: sha1(x.encode()).digest()[:1], ng)) if isinstance(ng, tuple) else ng for ng in ( \
            [ngram for sg in self.subgraphs for ngram in sg.ngrams(n, across_nodes=across_nodes)] if all_subgraphs else \
            self.acyclic_graph.ngrams(n, across_nodes=across_nodes))])

