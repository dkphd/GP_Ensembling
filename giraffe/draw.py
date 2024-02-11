from graphviz import Digraph

from giraffe.node import ValueNode, Node
from giraffe.tree import Tree
import giraffe

import numpy as np

from typing import Union

def draw_tree(to_draw: Union[Tree, Node], dot=None, add_val_eval=True):

    if isinstance(to_draw, giraffe.tree.Tree):
        node = to_draw.root
    else:
        node = to_draw

    if dot is None:
        dot = Digraph(comment="Tree")

    if isinstance(node, ValueNode):
        if node.value is not None:
            value = node.value.numpy() if (np.prod(node.value.shape) <= 9) else f"Tensor with memory adress: {hex(id(node.value))}"
        else:
            value = None

        
        if node.evaluation is not None:
            evaluation = node.evaluation.numpy() if (np.prod(node.evaluation.shape) <= 9) else f"Tensor with memory adress: {hex(id(node.evaluation))}"
        else:
            evaluation = None

        display_string = f"Value Node\n"
        
        if node.id is not None:
            display_string += f"Model ID: {node.id}\n"

        if add_val_eval:
            display_string += f"Value: {value} | Eval: {evaluation}"

        dot.node(
            f"{hex(id(node))}",
            display_string,
        )
    else:
        dot.node(f"{hex(id(node))}", f"Op\n{type(node).__name__}")

    for child in node.children:
        draw_tree(child, dot, add_val_eval)
        dot.edge(f"{hex(id(node))}", f"{hex(id(child))}")

    return dot
