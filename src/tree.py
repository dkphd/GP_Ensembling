from tinygrad.tensor import Tensor

import numpy as np

import random

from src.node import *


class Tree:
    def __init__(self, root: ValueNode, mutation_chance=0.1):

        self.root = root
        self.nodes = {"value_nodes": [self.root], "op_nodes": []}
        self.mutation_chance = mutation_chance

    # factory methods
    @staticmethod
    def create_tree_from_models(models, mutation_chance = 0.1):
        idx = np.random.choice(len(models))
        root = ValueNode(None, [], models[idx], idx)
        tree = Tree(root, mutation_chance)
        return tree

    @staticmethod
    def create_random_tree(mutation_chance = 0.1):
        root = ValueNode(None, [], Tensor.randn(1, 2))
        tree = Tree(root, mutation_chance)
        return tree

    def create_tree_from_root(root: Node, mutation_chance = 0.1):
        tree =  Tree(root, mutation_chance)
        tree.update_nodes()
        return tree

    @property
    def evaluation(self):
        return self.root.evaluation if self.root.evaluation is not None else self.root.calculate()

    @property
    def nodes_count(self):
        return len(self.nodes["value_nodes"]) + len(self.nodes["op_nodes"])

    def recalculate(self):
        self._clean_evals()
        return self.evaluation

    def copy(self):
        return Tree.create_tree_from_root(self.root.copy_subtree())

    def prune_at(self, node: Node):
        if node.parent is None:
            raise Exception("Cannot prune root node")

        subtree_nodes = node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].remove(subtree_node)
            else:
                self.nodes["op_nodes"].remove(subtree_node)

        node.parent.children.remove(node)

        self._clean_evals()

    def append_after(self, node: Node, new_node: Node):
        subtree_nodes = new_node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].append(subtree_node)
            else:
                self.nodes["op_nodes"].append(subtree_node)

        new_node.parent = node
        node.children.append(new_node)

        self._clean_evals()

    def replace_at(self, at: Node, replacement: Node):

        at_parent = at.parent
        
        if at_parent is None:
            print("Warning: node at replacement is root node")
            self.root = replacement
        else:
            at_parent.children.remove(at)
            replacement.parent = at_parent
            at_parent.children.append(replacement)

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

    def update_nodes(self):
        self.nodes = {"value_nodes": [], "op_nodes": []}
        for node in self.root.get_nodes():
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)


    def _clean_evals(self):
        for node in self.nodes["value_nodes"]:
            node.evaluation = None

