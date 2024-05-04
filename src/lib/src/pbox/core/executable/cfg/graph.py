# -*- coding: UTF-8 -*-
import networkx.classes.graph as ncg
from tinyscript import functools
from tinyscript.helpers import zeropad

from ....helpers.figure import plt


_DEFAULT_EXCLUDE = set()


# -------------------------------------- ADD-ONS TO networkx.classes.graph.Graph ---------------------------------------
def __getitem__(self, name):
    if not isinstance(name, str):
        return self.adj[name]
    if not name.startswith("_"):
        return getattr(self, name)
    raise KeyNotAllowedError(name)
ncg.Graph.__getitem__ = __getitem__


def draw(self, **kw):
    import networkx as nx
    pos = nx.spring_layout(self)
    nx.draw_networkx_nodes(self, pos, node_color="blue", node_size=12)
    ekw = {'arrows': True, 'arrowsize': 4} if isinstance(self, nx.classes.digraph.DiGraph) else {}
    nx.draw_networkx_edges(self, pos, edgelist=self.edges, edge_color="black", node_size=12, **ekw)
    plt.savefig("graph.png", **{k: kw.get(k, config[k]) for k in ["bbox_inches", "dpi", "format"]})
    plt.clf()
ncg.Graph.draw = draw


@staticmethod
def filter_nodes(nodes, exclude):
    exclude = [node.signature for node in exclude]    
    return [n for n in nodes if n.signature not in exclude]
ncg.Graph.filter_nodes = filter_nodes


def find_root(self, node, exclude=_DEFAULT_EXCLUDE):
    """ Get the best candidate for a (sub_)root_node by traversing the graph upward, ensuring the input node is still
         part of the subgraph extracted from the returned (sub_)root_node.
    
    :param node:    CFG node extracted by Angr
    :param exclude: set of node signatures to exclude in the extracted subgraph
    :return:        best candidate for a (sub_)root_node by traversing the graph upward, ensuring the input node is
                     still part of the subgraph extracted from the returned (sub_)root_node
    """
    visited, sub_root_node, already_checked_nodes = set(), node, [node]
    while self.predecessors(sub_root_node):
        s1 = sub_root_node.signature
        sub_root_node = list(self.predecessors(sub_root_node))[0]
        s2 = sub_root_node.signature
        if s1 == s2 or s2 in visited or s2 in exclude:
            break
        valid, already_checked_nodes = valid_sub_root_node(sub_root_node, already_checked_nodes)
        if not valid:
            break
        visited.add(s1)
    return sub_root_node
ncg.Graph.find_root = find_root


def iter_nodes(self, exclude=_DEFAULT_EXCLUDE):
    """ Iterate over nodes downwards from the provided root_node """
    queue, visited = [self.root_node], set()
    while queue:
        node = queue.pop(0)
        yield node
        visited.add(node.signature)
        for successor in node.successors:
            s = successor.signature
            if s in visited or s in exclude:
                continue
            queue.append(successor)
ncg.Graph.iter_nodes = iter_nodes


def num_neighbors(self, node=None):
    """ Get the number of successors and predecessors of a targeted node. """
    ns, np = len(list(self.successors(node))), len(list(self.predecessors(node)))
    if config['include_cut_edges'] and node.irsb:
        if node.irsb[1]:
            ns += len(node.irsb[1])
        if node.irsb[0]:
            np += node.irsb[0]
    return ns, np
ncg.Graph.num_neighbors = num_neighbors


@functools.lru_cache
def ngrams(self, n, length=None, across_nodes=True):
    """ Gets a list of ngrams. """
    #FIXME: in some cases, not enough ngrams to reach 'length' => shall pad with '-1'
    ngrams = []
    _extend = lambda bs: ngrams.extend([bs[i:i+n] for i in range(len(bs) - n + 1)])
    _fmt = lambda bs, pb: (type(bs)() if pb is None else pb) + bs
    # recursive function for getting ngrams according to the across_nodes method (includes the ngrams that consist
    #  of instuctions of connected nodes)
    def _get_ngrams_across_nodes(node, pre_bytes, recursion_limit):
        if length and len(ngrams) >= length or recursion_limit <= 50:
            return
        if node.byte_string:
            bstr = _fmt(node.byte_string, pre_bytes)
            _extend(bstr)
            pre_bytes = bstr[-n+1:]
        if node.irsb and node.irsb[1]:
            for cut_successor_bstr in node.irsb[1]:
                if cut_successor_bstr:
                    _extend(_fmt(cut_successor_bstr, pre_bytes))
        for successor in self.successors(node):
            _get_ngrams_across_nodes(successor, pre_bytes, recursion_limit-1)
    # get ngrams depending on the chosen method
    if across_nodes:
        _get_ngrams_across_nodes(self.root_node, None, RECURSION_LIMIT)
    else:
        queue, visited, mnemonics = [self.root_node], set(), isinstance(self.root_node.byte_string, tuple)
        _pad = lambda bs: bs + ('', ) * (n - len(bs)) if mnemonics else bs.ljust(n, b'\0')
        while queue:
            if length and len(ngrams) >= length:
                break
            node = queue.pop(0)
            if node.byte_string:
                _extend(_pad(node.byte_string))
            if node.irsb and node.irsb[1]:
                for cut_successor_bstr in node.irsb[1]:
                    if cut_successor_bstr:
                        _extend(_pad(cut_successor_bstr))
            for successor in self.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)
    return ngrams[:length] if length else ngrams
ncg.Graph.ngrams = ngrams


@functools.lru_cache
def signature(self, length, exact=True):
    set_depth(self)
    signature, queue, visited = [], [self.root_node], set()
    self.root_node.soot_block['idx'] = i = 1
    if exact:
        while queue and i < length:
            node = queue.pop(0)
            visited.add(node)
            for successor in sorted(self.successors(node), key=lambda n: -n.soot_block['depth']):
                if successor not in visited:
                    successor.soot_block['idx'] = i = i+1
                    queue.append(successor)
        queue = [self.root_node]
        while queue and len(signature) < length:
            node = queue.pop(0)
            for j, successor in enumerate(sorted(self.successors(node), key=lambda n: -n.soot_block['depth'])):
                signature.append(successor.soot_block['idx'])
                queue.append(successor)
            for _ in range(1-j):
                signature.append(0)
    else:
        while queue and len(signature) < length:
            node = queue.pop(0)
            visited.add(node)
            # 'approximate' signature
            # Reference: https://ieeexplore.ieee.org/document/8170793
            ns, np = self.num_neighbors(node)
            signature.append((ns << 6) | min(np, 63))
            for successor in sorted(self.successors(node), key=lambda n: -n.soot_block['depth']):
                if successor not in visited:
                    successor.soot_block['idx'] = i = i+1
                    queue.append(successor)
    return zeropad(length, default=0)(signature)
ncg.Graph.signature = signature


# ---------------------------------------------- LOCALLY USED FUNCTIONS ------------------------------------------------
def set_depth(graph):
    """ Set the depth parameter for each node in the input graph downwards connected to the (sub_)root_node. """
    niter = config['depth_max_iterations']
    def _set_depth(node, limit):
        nonlocal niter
        if niter <= 0 or limit <= 50:
            return 0
        if not node.size:
            node.size = 0
        max_successor_depth = 0
        for successor in graph.successors(node):
            max_successor_depth = max(max_successor_depth, _set_depth(successor, limit-1))
        node.soot_block = {'depth': node.size + max_successor_depth, 'idx': None}
        niter -= 1
        return node.soot_block['depth']
    root_depth = (getattr(graph.root_node, "soot_block") or {}).get('depth') or _set_depth(graph.root_node, RECURSION_LIMIT)
    for node in graph.nodes:
        if node.soot_block is None:
            node.soot_block = {'depth': -1, 'idx': None}
    return root_depth


def valid_sub_root_node(sub_root_node, already_checked_nodes):
    """ Verifies if using this sub_root_node to extract a subgraph will yield a subgraph that contains the original node
         from which this sub_root_node was calculated (with get_root_node(node))
    
    :param sub_root_node:         CFG node extracted by Angr
    :param already_checked_nodes: list of CFG nodes extracted by Angr containing the nodes that have already been
                                   visited when getting the downward subgraph connected to a successor of sub_root_node
    :return: (boolean indicating whether this sub_root_node can be considered,
              list of CFG nodes extracted by Angr containing the nodes that have already been visited when getting the
               downward subgraph connected to a successor of sub_root_node)
    """
    sub_root_successor, visited = [], set()
    # filter out duplicate nodes
    for node in sub_root_node.successors:
        s = node.signature
        if s not in visited:
            visited.add(s)
            sub_root_successor.append(node)
    if len(sub_root_successor) > 1:
        sub_root_successor = filter_nodes(sub_root_successor, [sub_root_node])
    if len(sub_root_successor) > 1:
        sub_root_successor = filter_nodes(sub_root_successor, already_checked_nodes)
    if len(sub_root_successor) > 1:
        sub_root_successor = filter_nodes(sub_root_successor,
                                          [n for n in sub_root_successor if n.addr + n.size != sub_root_node.addr])
    l = len(sub_root_successor)
    if l == 0:
        return False, []
    if l > 1:
        raise ValueError("No valid subroot node found")
    subgraph = list(iter_nodes(sub_root_successor[0]))
    new_nodes = filter_nodes(subgraph, already_checked_nodes)
    if len(subgraph) == len(new_nodes):
        already_checked_nodes.append(sub_root_node)
        already_checked_nodes.extend(new_nodes)
        return True, already_checked_nodes
    else:
        return False, []

