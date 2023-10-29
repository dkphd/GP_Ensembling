from src.tree import Tree
from src.node import *

from copy import deepcopy

import numpy as np

def crossover(tree1: Tree, tree2: Tree, debug=False, mutation_chance_crossover=False):

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

    if mutation_chance_crossover:
        mut_chances = [tree1.mutation_chance, tree2.mutation_chance]
        left_bound = np.clip(0.9*np.min(mut_chances), 0, 1)
        right_bound = np.clip(1.1*np.max(mut_chances), 0, 1)
        tree1.mutation_chance, tree2.mutation_chance = np.random.uniform(left_bound, right_bound, 2)


    if debug:
        if isinstance(node1, ValueNode):
            print(f"Crossover performed at node with id {node1.id} with value {node1.value.numpy()} and node with id {node2.id} with value {node2.value.numpy()}")
        else:
            print(f"Crossover performed at node {node1} and node {node2}")

    return tree1, tree2


def append_new_node_mutation(tree: Tree, models, debug=False):
    tree = tree.copy()

    idx_model = np.random.randint(len(models))

    node = tree.get_random_node()
    if isinstance(node, ValueNode):
        new_op = MeanNode(node, []) # TODO: randomize operator
        new_val = ValueNode(new_op, [], models[idx_model], idx_model)
        new_op.add_child(new_val)
        tree.append_after(node, new_op)
    else:
        new_val = ValueNode(None, [], models[idx_model], idx_model)
        tree.append_after(node, new_val)

    tree.recalculate()

    if debug:
        if isinstance(node, ValueNode):
            print(f"Append mutation performed at node with id {node.id} with value {node.value.numpy()}")
        else:
            print(f"Append mutation performed at node {node}")

    return tree

def lose_branch_mutation(tree: Tree):

    tree = tree.copy()
    node = tree.get_random_node(allow_root=False)
    tree.prune_at(node)

    return tree
