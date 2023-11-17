from argparse import ArgumentParser
from pathlib import Path

from src.tree import Tree
from src.draw import draw_tree
from src.gp_ops import *
from src.globals import DEBUG

from tinygrad.tensor import Tensor
from torch import load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from paretoset import paretoset

np.random.seed(1992)

def load_torch_to_tinygrad(input_path: Path):
    paths = [path.name for path in input_path.glob("*.pt")]
    tensors = [Tensor(load(input_path / path).numpy()) for path in paths]
    return np.array(paths), tensors


def f1_score_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    pred = (pred > 0.5).float()
    gt = gt.float()
    return f1_score(gt.numpy(), pred.numpy()) 


def first_uniques(arr):

    mask = []
    for index, item in enumerate(arr):
        if item not in arr[:index]:
            mask.append(True)
        else:
            mask.append(False)

    return mask
        

def choose_pareto_optimal(population, fitnesses):
    sizes = np.array([tree.nodes_count for tree in population])
    df = pd.DataFrame({"fitness": fitnesses, "size": sizes})
    mask = paretoset(df, sense=["max", "min"])
    if DEBUG:
        plt.figure(figsize=(6,6))
        #draw selected pareto optimal points in red and the rest in blue
        plt.scatter(df.fitness[~mask], df['size'][~mask], c="b", label="non-pareto-optimal")
        plt.scatter(df.fitness[mask], df['size'][mask], c="r", label="pareto-optimal")
        plt.legend()
        plt.xlabel("fitness")
        plt.ylabel("number of nodes")
        plt.savefig('./pareto.png')

    population = [population[idx] for idx in np.where(mask)[0]]
    fitnesses = fitnesses[mask]
    return population, fitnesses


def choose_sorted(population, fitnesses, n):
    fitnesses_sorted = np.argsort(fitnesses)
    fitnesses = fitnesses[fitnesses_sorted[-n:]]
    population = [population[idx] for idx in fitnesses_sorted[-population_size:]]
    return population, fitnesses


def regenerate_population(population, fitnesses, n):
    while len(population) < n:
        new_tree = Tree.create_tree_from_models(tensors, ids = paths)
        code = new_tree.__repr__()
        if code not in population_codes:
            population.append(new_tree)
            population_codes.append(code)
            fitnesses = np.concatenate([fitnesses, [f1_score_fitness(new_tree, gt)]])

    return population, fitnesses
        
       
def load_args():

    parser = ArgumentParser()
    parser.add_argument("--input_path", default="./test_probs")
    parser.add_argument("--gt_path", default="./test_y.pt")
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--population_multplier", type=int, default=1)
    parser.add_argument("--tournament_size", type=int, default=5)
    args = parser.parse_args()

    return Path(args.input_path), Path(args.gt_path), args.population_size, args.population_multplier, args.tournament_size

if __name__ == "__main__":
    
    in_path, gt_path, population_size, population_multplier, tournament_size = load_args()

    paths, tensors = load_torch_to_tinygrad(in_path)

    gt = Tensor(load(gt_path).numpy())

    print("Generating population...")

    population = [Tree.create_tree_from_models(tensors, ids = paths) for _ in range(population_size)]
    

    ids = np.arange(population_size)
    addional_population = []

    print("Calculating fitnesses...")

    print("Starting evolution...")

    for i in range(30):

        fitnesses = np.array([f1_score_fitness(tree, gt) for tree in population])

        while len(addional_population) < len(population) * population_multplier:
            idxs = np.random.choice(ids, tournament_size, replace=False)
            tournament_fitnesses = np.array(fitnesses)[idxs]
            parent_idxs = idxs[np.argsort(tournament_fitnesses)[-2:]]
            parent1, parent2 = population[parent_idxs[0]], population[parent_idxs[1]]

            try:
                child1, child2 = crossover(parent1, parent2, mutation_chance_crossover=False, debug=DEBUG)
            except Exception as e:
                print("Crossover failed due to: ", e)
                continue

            child1.update_nodes()
            child1._scan_nodes_for_lack_of_parent()
            child2.update_nodes()
            child2._scan_nodes_for_lack_of_parent()

            addional_population += [child1, child2]

        print("Mutating...")

        #mutations
        for tree in population:
            if np.random.rand() < tree.mutation_chance:
                mutated_tree = append_new_node_mutation(tree, tensors, ids = paths)
                mutated_tree.update_nodes()
                mutated_tree._scan_nodes_for_lack_of_parent()
                addional_population.append(mutated_tree)

        population += addional_population
        additional_fitnesses = np.array([f1_score_fitness(tree, gt) for tree in addional_population])
        fitnesses = np.concatenate([fitnesses, additional_fitnesses])

        addional_population = []

        # At this place we remove duplicates and regenerate the population
        population_codes = [tree.__repr__() for tree in population]
        unique_codes = first_uniques(population_codes)

        print("Removing duplicates...")
        print(f"There is {len(population_codes) - sum(unique_codes)} duplicates in population")    

        population = [tree for tree, unique in zip(population, unique_codes) if unique]
        fitnesses = fitnesses[unique_codes]

        # regenearte population
        population, fitnesses = regenerate_population(population, fitnesses, population_size)

        population, fitnesses = choose_pareto_optimal(population, fitnesses)

        print(F"Population size: {len(population)} after selection and before regeneration")

        # regenearte population
        population, fitnesses = regenerate_population(population, fitnesses, population_size)


        print("Best fitness:")

        print(np.max(fitnesses))
        dot = draw_tree(population[np.argmax(fitnesses)])
        dot.render(f"trees/best_{i}", view=True, format='png')

    print(fitnesses)
