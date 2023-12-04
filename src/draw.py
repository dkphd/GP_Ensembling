from graphviz import Digraph

from src.node import ValueNode, Node
from src.tree import Tree

import numpy as np

from typing import Union

def draw_tree(to_draw: Union[Tree, Node], dot=None):

    if isinstance(to_draw, Tree):
        node = to_draw.root
    else:
        node = to_draw

    if dot is None:
        dot = Digraph(comment="Tree")

    if isinstance(node, ValueNode):
        value = node.value.numpy() if (np.prod(node.value.shape) <= 9) else f"Tensor with memory adress: {hex(id(node.value))}"

        
        if node.evaluation is not None:
            evaluation = node.evaluation.numpy() if (np.prod(node.evaluation.shape) <= 9) else f"Tensor with memory adress: {hex(id(node.evaluation))}"
        else:
            evaluation = None

        display_string = f"Value Node\n"
        if node.id is not None:
            display_string += f"Model ID: {node.id}\n"
        display_string += f"Value: {value} | Eval: {evaluation}"

        dot.node(
            f"{hex(id(node))}",
            display_string,
        )
    else:
        dot.node(f"{hex(id(node))}", f"Op\n{type(node).__name__}")

    for child in node.children:
        draw_tree(child, dot)
        dot.edge(f"{hex(id(node))}", f"{hex(id(child))}")

    return dot
