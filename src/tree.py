from tinygrad.tensor import Tensor

import numpy as np
from functools import reduce
from operator import add, sub, mul, truediv
import random

from graphviz import Digraph

from functools import partial


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
