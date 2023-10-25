from graphviz import Digraph

from src.node import ValueNode


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
