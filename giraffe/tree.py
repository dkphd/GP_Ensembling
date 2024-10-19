from giraffe.globals import DEVICE
from giraffe.globals import BACKEND as B
from giraffe.types import Tensor

import numpy as np


from giraffe.node import OperatorNode, ValueNode, Node
from giraffe.globals import VERBOSE
from giraffe.utils import Pickle


class Tree:
    def __init__(self, root: ValueNode, mutation_chance=0.1):
        self.root = root

        if isinstance(self.root, OperatorNode):
            raise Exception("Cannot get evaluation of tree with OpNode as root")

        self.nodes = {"value_nodes": [self.root], "op_nodes": []}
        self.mutation_chance = mutation_chance

    # factory methods
    @staticmethod
    def create_tree_from_tensors(tensors, mutation_chance=0.1, ids=None):
        if ids is None:
            ids = np.arange(len(tensors))
        assert len(tensors) == len(ids)

        idx = np.random.choice(len(tensors))
        root = ValueNode(None, [], tensors[idx], ids[idx])
        tree = Tree(root, mutation_chance)
        return tree

    @staticmethod
    def create_random_tree(mutation_chance=0.1):
        root = ValueNode(None, [], Tensor.randn(1, 2))
        tree = Tree(root, mutation_chance)
        return tree

    @staticmethod
    def create_tree_from_root(root: Node, mutation_chance=0.1):
        tree = Tree(root, mutation_chance)
        tree.update_nodes()
        return tree

    @property
    def evaluation(self):
        return B.squeeze(self.root.evaluation if self.root.evaluation is not None else self.root.calculate())

    @property
    def nodes_count(self):
        return len(self.nodes["value_nodes"]) + len(self.nodes["op_nodes"])

    @property
    def top_sorted_nodes(
        self,
    ):  # should use Node's get_nodes() which needs to return top_sorted nodes instead (as in calculate)
        pass  # TODO

    def recalculate(self):
        self._clean_evals()
        return self.evaluation

    def copy(self):
        return Tree.create_tree_from_root(self.root.copy_subtree())

    def prune_at(self, node: Node):  # remove node from the tree along with its children
        if node.parent is None:
            raise Exception("Cannot prune root node")
        if isinstance(node.parent, OperatorNode) and (len(node.parent.children) < 2):
            return self.prune_at(node.parent)

        subtree_nodes = node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].remove(subtree_node)
            else:
                self.nodes["op_nodes"].remove(subtree_node)

        node.parent.remove_child(node)

        self._clean_evals()

    def append_after(self, node: Node, new_node: Node):
        subtree_nodes = new_node.get_nodes()

        for subtree_node in subtree_nodes:
            if isinstance(subtree_node, ValueNode):
                self.nodes["value_nodes"].append(subtree_node)
            else:
                self.nodes["op_nodes"].append(subtree_node)

        new_node.parent = node
        node.add_child(new_node)

        self._clean_evals()

    def replace_at(
        self, at: Node, replacement: Node
    ):  # like prune at and then append after parent, but without parameters adjustment (may be worth it to reimplement)
        at_parent = at.parent

        if at_parent is None:
            assert isinstance(self.root, ValueNode), "Root must be a value node"
            if VERBOSE >= 3:
                print("Warning: node at replacement is root node")
            self.root = replacement
        else:
            at_parent.replace_child(at, replacement)

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
        np.random.shuffle(order)
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

    def save_tree_architecture(self, output_path):  # TODO: needs adjustment for weighted node
        copy_tree = self.copy()
        for index_node, node in enumerate(copy_tree.nodes["value_nodes"]):
            node.value = node.evaluation = None

        Pickle.save(output_path, copy_tree)

    @staticmethod
    def load_tree_architecture(architecture_path):  # TODO: needs adjustmed for weighted node
        architeture = Pickle.load(architecture_path)
        for operator_node in architeture.nodes["op_nodes"]:
            operator_node.operator = type(operator_node).get_operator()
        return architeture

    @staticmethod
    def load_tree(architecture_path, preds_directory, tensors={}):  # NEEDS TO USE BACKEND
        current_tensors = {}
        current_tensors.update(
            tensors
        )  # needed because otherwise the tensors would be update in place in class and load tree would not reload
        loaded = Tree.load_tree_architecture(architecture_path)
        if VERBOSE:
            unique_ids = []
        for value_node in loaded.nodes["value_nodes"]:
            node_id = value_node.id
            if VERBOSE and node_id not in unique_ids:
                unique_ids.append(node_id)
            if node_id not in current_tensors:
                current_tensors[node_id] = B.load_torch(preds_directory / node_id, DEVICE)
            value_node.value = current_tensors[node_id]

        if VERBOSE:
            print(f"Loaded tree has {len(unique_ids)} unique ids")
            print(f"Loaded tree has {len(loaded.nodes['value_nodes'])} value nodes")

        return loaded, current_tensors

    def _clean_evals(self):
        for node in self.nodes["value_nodes"]:
            node.evaluation = None

    def _scan_nodes_for_lack_of_parent(self):
        for node in self.nodes["op_nodes"]:
            if node.parent is None:
                raise Exception("I have no parent!")

    def __repr__(self):
        return "_".join(node.code for node in self.root.get_nodes())
