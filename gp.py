from argparse import ArgumentParser
from pathlib import Path

from src.tree import Tree
from src.draw import draw_tree
from src.gp_ops import *
from src.globals import DEBUG

from tinygrad.tensor import Tensor
from torch import load

import numpy as np

from sklearn.metrics import f1_score

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


def load_args():

    parser = ArgumentParser()
    parser.add_argument("--input_path", default="./train_probs")
    parser.add_argument("--gt_path", default="./train_y.pt")
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

    fitnesses = [f1_score_fitness(tree, gt) for tree in population]

    print("Starting evolution...")

    for i in range(30):

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
        fitnesses += [f1_score_fitness(tree, gt) for tree in addional_population]
        addional_population = []
        fitnesses_sorted = np.argsort(fitnesses)
        fitnesses = list(np.array(fitnesses)[fitnesses_sorted[-population_size:]])
        population = [population[idx] for idx in fitnesses_sorted[-population_size:]]

        print("Best fitness:")

        print(np.max(fitnesses))
        dot = draw_tree(population[-1])
        dot.render(f"trees/best_{i}", view=True, format='png')

    print(fitnesses)
