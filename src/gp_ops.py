from src.tree import Tree
from src.node import *

from copy import deepcopy

def crossover(tree1: Tree, tree2: Tree):

    tree1, tree2 = deepcopy(tree1), deepcopy(tree2)

    node1 = tree1.get_random_node()
    if isinstance(node1, ValueNode):
        node2 = tree2.get_random_node("value_nodes")
    else:
        node2 = tree2.get_random_node("op_nodes")

    tree1.replace_at(node1, node2)
    tree2.replace_at(node2, node1)

    tree1.recalculate()
    tree2.recalculate()

    return tree1, tree2