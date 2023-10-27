from graphviz import Digraph

from src.node import ValueNode

import numpy as np

def draw_tree(node, dot=None):
    if dot is None:
        dot = Digraph(comment="Tree")

    if isinstance(node, ValueNode):
        value = node.value.numpy() if (np.prod(node.value.shape) <= 9) else f"Tensor with memory adress: {id(node.value)}"

        
        if node.evaluation is not None:
            evaluation = value = node.evaluation.numpy() if (np.prod(node.evaluation.shape) <= 9) else f"Tensor with memory adress: {id(node.evaluation)}"
        else:
            evaluation = None


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
