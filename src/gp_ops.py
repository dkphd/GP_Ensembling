from src.tree import Tree
from src.node import *

from copy import deepcopy

def crossover(tree1: Tree, tree2: Tree, debug=False):

    tree1, tree2 = tree1.copy(), tree2.copy()

    node1 = tree1.get_random_node()
    if isinstance(node1, ValueNode):
        node2 = tree2.get_random_node("value_nodes")
    else:
        node2 = tree2.get_random_node("op_nodes")

    tree1.replace_at(node1, node2.copy_subtree())
    tree2.replace_at(node2, node1.copy_subtree())

    tree1.recalculate()
    tree2.recalculate()

    if debug:
        if isinstance(node1, ValueNode):
            print(f"Crossover performed at node with id {node1.id} with value {node1.value.numpy()} and node with id {node2.id} with value {node2.value.numpy()}")
        else:
            print(f"Crossover performed at node {node1} and node {node2}")

    return tree1, tree2