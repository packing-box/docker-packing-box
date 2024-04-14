# -*- coding: UTF-8 -*-
import angr.knowledge_plugins.cfg.cfg_node as akc

from ....helpers.mixins import GetItemMixin


akc.CFGNode.__getitem__ = GetItemMixin.__getitem__


@property
def neighbors(self):
    """ Get the number of successors and predecessors. """
    ns, np = len(self.successors), len(self.predecessors)
    if config['include_cut_edges'] and self.irsb:
        if node.irsb[1]:
            ns += len(node.irsb[1])
        if node.irsb[0]:
            np += node.irsb[0]
    return ns, np
akc.CFGNode.neighbors = neighbors


@property
def signature(self):
    return tuple(tuple(n.addr for n in l) for l in (self.predecessors, [self], self.successors))
akc.CFGNode.signature = signature

