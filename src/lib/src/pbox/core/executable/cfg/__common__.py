# -*- coding: UTF-8 -*-
from tinyscript import code, functools, hashlib, logging, re
from tinyscript.helpers import Capture, Timeout

from ....helpers.data import get_data
from ....helpers.mixins import *


__all__ = ["CFG"]

def __init_common_api_imports():
    return {grp: set(functools.reduce(lambda l1, l2: l1 + l2, get_data(grp).get('COMMON_DLL_IMPORTS', {}).values() \
            or [()])) for grp in FORMATS.keys() if grp != "All"}
_COMMON_API_IMPORTS = lazy_load_object("_COMMON_API_IMPORTS", __init_common_api_imports)
def __init_common_malicious_apis():
    return {grp: set(get_data(grp).get('COMMON_MALICIOUS_APIS', {})) for grp in FORMATS.keys() if grp != "All"}
_COMMON_MALICIOUS_APIS = lazy_load_object("_COMMON_MALICIOUS_APIS", __init_common_malicious_apis)
_DEFAULT_EXCLUDE = set()

_sha1 = lambda x: hashlib.sha1(x.encode()).digest()


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


def _sorted_hist(vector):
    """ Orders the elements of the input list first by occurence, then by value, yielding a sorted histogram.
    
    :param vector: a list of unordered elements
    :return: (list of ordered elements; first by occurence; then by value,
              list of number of occurences of the elements in ordered list)
    """
    from collections import Counter
    return list(map(list, zip(*sorted(Counter(vector).items(), key=lambda x: (x[1], x[0]), reverse=True))))


class CFG(GetItemMixin, ResetCachedPropertiesMixin):
    engines = {k: getattr(angr.engines, "UberEngine" if k in ["default", "vex"] else f"UberEngine{k.capitalize()}") \
               for k in ANGR_ENGINES}
    logger = logging.getLogger("executable.cfg")
    
    def __init__(self, target, engine=None, **kw):
        self.__target = target
        try:
            self.__project = angr.Project(str(self.__target), load_options={'auto_load_libs': False},
                                          engine=self.engines[engine or config['angr_engine']])
            self.model = self.__project.kb.cfgs.new_model(f"{self.__target}")
        except Exception as e:
            self.__project = self.model = None
            self.logger.exception(e)
    
    def __filter(self, d):
        return d.get(self.__target.format, {}) or d.get(self.__target.group, {})
    
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
    
    def indirect_jumps(self):
        for m, o in self.iterinsns():
             if m in X86_64_JUMP_MNEMONICS and not o.startswith('0x'):
                yield m
    
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
    
    def jumps(self):
        for m, _ in self.iterinsns():
            if m in X86_64_JUMP_MNEMONICS:
                yield m
    
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
    
    # ------------------------------------------ LAZY ONE-TIME COMPUTATIONS --------------------------------------------
    @cached_property
    def acyclic_graph(self):
        """ Compute and return the acyclic version of the CFG. """
        return self.__class__.to_acyclic(self.graph)
    
    @cached_property
    def common_imports(self):
        return self.imported_apis & self.__filter(_COMMON_API_IMPORTS)
    
    @cached_property
    def imported_apis(self):
        if not self.model:
            self.compute()
        return set(self.model.project.loader.main_object.imports.keys())
    
    @cached_property
    def malicious_imported_apis(self):
        return self.imported_apis & self.__filter(_COMMON_MALICIOUS_APIS)
    
    @cached_property
    def malicious_used_apis(self):
        return self.used_apis & self.__filter(_COMMON_MALICIOUS_APIS)
    
    @functools.lru_cache
    def ngram_histogram(self, n, across_nodes=True, all_subgraphs=False):
        """ Gets a sorted histogram of ngrams. """
        return _sorted_hist([b''.join(map(lambda x: _sha1(x)[:1], ng)) if isinstance(ng, tuple) else ng for ng in \
                             ([ngram for sg in self.subgraphs for ngram in sg.ngrams(n, across_nodes=across_nodes)] \
                             if all_subgraphs else self.acyclic_graph.ngrams(n, across_nodes=across_nodes))])
    
    @cached_property
    def nodesize_histogram(self):
        """ Gets a sorted histogram of node sizes. """
        return _sorted_hist([node.size if node.size else 0 for node in self.nodes])
    
    @cached_property
    def register_type_counts(self):
        reg_counts = {group: 0 for group in X86_64_REGISTERS}
        for token in sum((re.sub(r"[,\+\*\[\]\-]", " ", it[1]).split() for it in self.iterinsns()), []):
            for group, reg_set in X86_64_REGISTERS.items():
                reg_counts[group] += token in reg_set
        return list(reg_counts.values())
    
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
    def used_apis(self):
        return {n.name for n in self.nodes if n.name is not None and n.name != "PathTerminator"}

