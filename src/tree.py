from tinygrad.tensor import Tensor

import numpy as np
from functools import reduce
from operator import add, sub, mul, truediv
import random

from graphviz import Digraph

from functools import partial


class Node:
    def __init__(self, parent=None, children=[], tree=None):
        self.parent = parent
        self.children = children  # child nodes
        self.tree = tree

    def add_child(self, child_node):
        self.children.append(child_node)


class OperatorNode(Node):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.parent = parent
        self.children = children

    def calculate(self):
        pass

    def __str__(self):
        return f"OperatorNode: {None}"


class BinaryOperatorNode(OperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = lambda x: None

    def calculate(self):
        return reduce(
            lambda x, y: self.operator(x, y),
            [self.parent.evaluation if self.parent.evaluation is not None else self.parent.value]
            + [child.calculate() for child in self.children],
        )

    def __str__(self):
        return f"BinaryOperatorNode: {self.operator}"


class ReductionOperatorNode(OperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = lambda x: None

    def calculate(self):
        parent_eval = self.parent.evaluation if self.parent.evaluation is not None else self.parent.value

        concat = Tensor.stack([parent_eval] + [child.calculate() for child in self.children], dim=0)
        return self.operator(concat)


class MeanNode(ReductionOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.mean, axis=0)

    def __str__(self):
        return f"MeanNode"


class WeightedMeanNode(MeanNode):
    def __init__(self, parent, children, tree, weights: Tensor):
        super().__init__(parent, children, tree)

        assert len(weights.shape) == 2
        assert weights.shape[0] == 1
        assert weights.shape[1] == len(children) + 1

        self.weights = weights
        self.operator = lambda x: super().operator(x * self.weights)


class AddNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = add


class SubNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = sub


class MulNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = mul


class DivNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = truediv


class ValueNode(Node):
    def __init__(self, parent, children, tree, value):
        super().__init__(parent, children, tree)
        self.value = value
        self.evaluation = None

    def calculate(self):
        if self.children:
            self.evaluation = self.children[0].calculate()
        else:
            self.evaluation = self.value
        return self.evaluation

    def __str__(self):
        return f"ValueNode: {self.value}"

    def add_child(self, child_node):
        super().add_child(child_node)
        self.evaluation = None


class Tree:
    def __init__(self, models=None):
        if models is None:
            self.models = [Tensor.randn(1, 2) for _ in range(10)]
            self.debug = True
        else:
            self.models = models
            self.debug = False

        self.root = ValueNode(None, [], self, np.random.choice(self.models))
        self.nodes = [self.root]

    def grow_tree(self, node, depth=0, max_depth=3):
        if depth >= max_depth:
            return

        # Add operator
        op_node = random.choice([AddNode, SubNode, MulNode, DivNode, WeightedMeanNode])
        if isinstance(op_node, WeightedMeanNode):
            op_node = op_node(node, [], self, weight_top=random.random())
        else:
            op_node = op_node(node, [], self)

        node.add_child(op_node)
        self.nodes.append(op_node)

        # Determine the number of children for the operator (either 1 or 2)
        num_children = 2 if random.random() < 0.2 or depth <= 1 else 1  # 20% chance for 2 children

        for i in range(num_children):
            # Add model with a unique ID (for illustration)
            val = np.random.choice(self.models)
            model_node = ValueNode(op_node, [], self, val)
            op_node.add_child(model_node)

            self.nodes.append(model_node)

            # Recursively grow the tree
            self.grow_tree(model_node, depth + 1, max_depth)


def draw_tree(node, dot=None):
    if dot is None:
        dot = Digraph(comment="Tree")

    if isinstance(node, ValueNode):
        value = node.value.numpy() if node.tree.debug else f"Tensor with memory adress: {id(node.value)}"

        if node.tree.debug:
            if node.evaluation is not None:
                evaluation = node.evaluation.numpy()
            else:
                evaluation = None
        else:
            evaluation = f"Tensor with memory adress: {id(node.evaluation)}" if node.evaluation is not None else None

        dot.node(
            f"{id(node)}",
            f"Value Node\nValue: {value} | Eval: {evaluation}",
        )
    else:
        dot.node(f"{id(node)}", f"Op\n{node.operator}")

    for child in node.children:
        draw_tree(child, dot)
        dot.edge(f"{id(node)}", f"{id(child)}")

    return dot
