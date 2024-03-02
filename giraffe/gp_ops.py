from giraffe.tree import Tree
from giraffe.node import ValueNode, MeanNode, MaxNode, MinNode
from giraffe.globals import VERBOSE


import numpy as np


def crossover(tree1: Tree, tree2: Tree, verbose=False, mutation_chance_crossover=False):
    tree1, tree2 = tree1.copy(), tree2.copy()

    node1 = tree1.get_random_node()
    if isinstance(node1, ValueNode):
        node2 = tree2.get_random_node("value_nodes")
        if type(node1) != type(node2):
            raise Exception("Cannot crossover nodes of different types")
    else:
        node_types = "op_nodes"
        node2 = tree2.get_random_node(node_types)

    replacement_node1 = node2.copy_subtree()
    replacement_node2 = node1.copy_subtree()

    tree1.replace_at(node1, replacement_node1)
    tree2.replace_at(node2, replacement_node2)

    tree1.recalculate()
    tree2.recalculate()

    if mutation_chance_crossover:
        mut_chances = [tree1.mutation_chance, tree2.mutation_chance]
        left_bound = np.clip(0.9 * np.min(mut_chances), 0, 1)
        right_bound = np.clip(1.1 * np.max(mut_chances), 0, 1)
        tree1.mutation_chance, tree2.mutation_chance = np.random.uniform(left_bound, right_bound, 2)

    return tree1, tree2


# Mutations


def append_new_node_mutation(tree: Tree, models, ids=None, allow_all_ops=False, **kwargs):
    tree = tree.copy()

    if ids is None:
        ids = np.arange(len(models))

    idx_model = np.random.choice(np.arange(len(models)))

    node = tree.get_random_node()
    if isinstance(node, ValueNode):
        if allow_all_ops:
            new_op = np.random.choice([MeanNode, MaxNode, MinNode], 1)[0](node, [])
        else:
            new_op = MeanNode(node, [])
        new_val = ValueNode(new_op, [], models[idx_model], ids[idx_model])
        new_op.add_child(new_val)
        tree.append_after(node, new_op)
    else:
        new_val = ValueNode(None, [], models[idx_model], ids[idx_model])
        tree.append_after(node, new_val)

    return tree


def lose_branch_mutation(tree: Tree, **kwargs):
    tree = tree.copy()
    node = tree.get_random_node(allow_root=False, allow_leaves=False)
    tree.prune_at(node)

    return tree


MUTATION_FUNCTIONS = [append_new_node_mutation, lose_branch_mutation]


def mutate_population(population, tensors, ids, allow_all_ops=False):
    mutated_trees = []
    for tree in population:
        if np.random.rand() < tree.mutation_chance:
            try:
                mutation_function = np.random.choice(MUTATION_FUNCTIONS, 1)[0]
                mutated_tree = mutation_function(tree, models=tensors, ids=ids, allow_all_ops=allow_all_ops)
                mutated_tree.update_nodes()
                mutated_trees.append(mutated_tree)
            except Exception as e:
                if VERBOSE > 2:
                    print("Mutation failed due to: ", e)
                continue

    return mutated_trees


###


def regenerate_population(population, fitnesses, n, tensors, gt, ids, fitness_function, population_codes=None):
    """
    Regenerate a population of Tree objects until it reaches a specified size.

    This function iteratively generates new Tree objects based on provided tensors and ids.
    It ensures that each new Tree is unique within the current population by checking its code representation
    against existing ones. The fitness of each new Tree is evaluated using the provided fitness function
    and appended to the fitnesses list. The population is grown until it reaches the specified size n.

    Parameters:
    - population (list): A list of Tree objects representing the current population.
    - fitnesses (np.ndarray): An array of fitness values corresponding to each Tree in the population.
    - n (int): The desired population size.
    - tensors (list/tuple): Data structure containing tensors used to generate new Tree objects.
    - gt: Ground truth data against which the fitness of each Tree is evaluated.
    - ids (list/tuple): Identifiers used in the creation of new Tree objects.
    - fitness_function (function): A function that takes a Tree object and the ground truth data,
                                   and returns a fitness score.
    - population_codes (list, optional): A list of string representations of the Trees in the population.
                                         If None, it's generated from the current population.

    """

    if population_codes is None:
        population_codes = [tree.__repr__() for tree in population]
    while len(population) < n:
        new_tree = Tree.create_tree_from_models(tensors, ids=ids)
        code = new_tree.__repr__()
        if code not in population_codes:
            population.append(new_tree)
            population_codes.append(code)
            fitnesses = np.concatenate([fitnesses, [fitness_function(new_tree, gt)]])

    return population, fitnesses
