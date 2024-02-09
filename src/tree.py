from tinygrad.tensor import Tensor

import numpy as np

import torch

from src.node import *
from src.globals import VERBOSE
from src.utils import Pickle

class Tree:
    def __init__(self, root: ValueNode, mutation_chance=0.1):

        self.root = root

        if isinstance(self.root, OperatorNode):
            raise Exception("Cannot get evaluation of tree with OpNode as root")

        self.nodes = {"value_nodes": [self.root], "op_nodes": []}
        self.mutation_chance = mutation_chance

    # factory methods
    @staticmethod
    def create_tree_from_models(models, mutation_chance = 0.1, ids = None):
        if ids is None:
            ids = np.arange(len(models))
        assert len(models) == len(ids)


        idx = np.random.choice(len(models))
        root = ValueNode(None, [], models[idx], ids[idx])
        tree = Tree(root, mutation_chance)
        return tree

    @staticmethod
    def create_random_tree(mutation_chance = 0.1):
        root = ValueNode(None, [], Tensor.randn(1, 2))
        tree = Tree(root, mutation_chance)
        return tree

    @staticmethod
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
    

    @property
    def top_sorted_nodes(self):
        pass # TODO
    

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
            if VERBOSE >= 3:
                print("Warning: node at replacement is root node")
            self.root = replacement
            if isinstance(self.root, OperatorNode):
                raise Exception("Cannot get evaluation of tree with OpNode as root")

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

    def get_random_node(self, nodes_type: str = None, allow_root=True, allow_leaves=True):

        if self.root.children == []:
            if allow_root:
                if nodes_type is None or nodes_type == "value_nodes":
                    return self.root
                else:
                    raise Exception("Tree has only root node and nodes_type is not value_nodes")
            else:
                raise Exception("Tree has only root node and allow_root is set to False")



        if nodes_type is None:
            nodes_type = np.random.choice(["value_nodes", "op_nodes"])

        order = np.arange(len(self.nodes[nodes_type]))
        for i in order:
            node = self.nodes[nodes_type][i]
            if (allow_leaves or node.children != []) and (allow_root or node != self.root):
                return node
        raise Exception("No node found that complies to the constraints")


    def update_nodes(self):
        self.nodes = {"value_nodes": [], "op_nodes": []}
        for node in self.root.get_nodes():
            if isinstance(node, ValueNode):
                self.nodes["value_nodes"].append(node)
            else:
                self.nodes["op_nodes"].append(node)


    def get_unique_value_node_ids(self):
        return list(set([node.id for node in self.nodes["value_nodes"]]))


    def save_tree_architecture(self, output_path):

        copy_tree = self.copy()
        for index_node, node in enumerate(copy_tree.nodes["value_nodes"]):
            node.value = node.evaluation = None
        
        Pickle.save(output_path, copy_tree)


    @staticmethod
    def load_tree_architecture(architecture_path):
        return Pickle.load(architecture_path)


    @staticmethod
    def load_tree(architecture_path, preds_directory, tensors = {}):
        loaded = Pickle.load(architecture_path)

        if VERBOSE:
            unique_ids = []
        for value_node in loaded.nodes["value_nodes"]:
            node_id = value_node.id
            if VERBOSE and node_id not in unique_ids:
                unique_ids.append(node_id)
            if node_id not in tensors:
                value_tensor = torch.load(preds_directory / node_id)
                tensors[node_id] = value_tensor
            value_node.value = tensors[node_id]
        
        if VERBOSE:
            print(f"Loaded tree has {len(unique_ids)} unique ids")
            print(f"Loaded tree has {len(loaded.nodes['value_nodes'])} value nodes")
                  
        return loaded, tensors

    def _clean_evals(self):
        for node in self.nodes["value_nodes"]:
            node.evaluation = None


    def _scan_nodes_for_lack_of_parent(self):
        for node in self.nodes["op_nodes"]:
            if node.parent is None:
                raise Exception("I have no parent!")


    def __repr__(self):
        return '_'.join(node.code for node in self.root.get_nodes())