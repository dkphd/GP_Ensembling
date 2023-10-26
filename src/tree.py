from tinygrad.tensor import Tensor

import numpy as np

import random

from src.node import *


class Tree:
    def __init__(self, models=None):
        if models is None:
            self.models = [Tensor.randn(1, 2) for _ in range(10)]
            self.debug = True
        else:
            self.models = models
            self.debug = False

        self.root = ValueNode(None, [], self, np.random.choice(models))
        self.nodes = {"value_nodes": [self.root], "op_nodes": []}

    @property
    def evaluation(self):
        return self.root.evaluation if self.root.evaluation is not None else self.root.calculate()

    @property
    def nodes_count(self):
        return len(self.nodes["value_nodes"]) + len(self.nodes["op_nodes"])

    def recalculate(self):
        self._clean_evals()
        return self.evaluation

    def prune_at(self, node: Node):
        if node.parent is None:
            raise Exception("Cannot prune root node")

        subtree_nodes = node.get_nodes()

        for node in subtree_nodes:
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].remove(node)
            else:
                self.nodes["op_nodes"].remove(node)

        node.parent.children.remove(node)

        self._clean_evals()

    def append_after(self, node: Node):
        subtree_nodes = node.get_nodes()

        for node in subtree_nodes:
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)

        node.parent.children.append(node)

        self._clean_evals()

    def replace_at(self, at: Node, replacement: Node):

        at_parent = at.parent
        
        if at_parent is None:
            print("Warning: node at replacement is root node")
        else:
            at_parent.children.remove(at)
            replacement.parent = at_parent
            at_parent.children.append(replacement)

        replacement.children = at.children

        if isinstance(at, ValueNode):
            self.nodes["value_nodes"].remove(at)
            self.nodes["value_nodes"].append(replacement)
        else:
            self.nodes["op_nodes"].remove(at)
            self.nodes["op_nodes"].append(replacement)

        self._clean_evals()

    def get_random_node(self, node_type: str = None):
        if node_type is None:
            node_type = random.choice(["value_nodes", "op_nodes"])

        return random.choice(self.nodes[node_type])

    def _clean_evals(self):
        for node in self.nodes["value_nodes"]:
            node.evaluation = None
