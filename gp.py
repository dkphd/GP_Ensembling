from argparse import ArgumentParser
from pathlib import Path

from src.draw import draw_tree
from src.gp_ops import *
from src.globals import DEBUG, STATE
from src.fitness import FitnessFunction, calculate_fitnesses
from src.ops import *
from src.loaders import load_torch_preds_from_directory

from tinygrad.tensor import Tensor
from torch import load

import numpy as np

np.random.seed(1992)

def load_args():

    parser = ArgumentParser()
    parser.add_argument("--input_path", default="./additional_valid_probs")
    parser.add_argument("--gt_path", default="./to_evolution/additional_valid_y.pt")
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--population_multplier", type=int, default=1)
    parser.add_argument("--tournament_size", type=int, default=5)
    parser.add_argument("--fitness_function", type=str, choices=list(FitnessFunction), default=FitnessFunction.F1_SCORE)
    args = parser.parse_args()

    return Path(args.input_path), Path(args.gt_path), args.population_size, args.population_multplier, args.tournament_size, args.fitness_function

if __name__ == "__main__":
    
    in_path, gt_path, population_size, population_multplier, tournament_size, fitness_function = load_args()

    paths, tensors = load_torch_preds_from_directory(in_path)

    gt = Tensor(load(gt_path).numpy())

    print("Generating population...")

    population, _ = regenerate_population([], [], population_size, tensors, gt, paths, fitness_function, population_codes = None)

    ids = np.arange(population_size)
    additional_population = []

    print("Calculating fitnesses...")

    print("Starting evolution...")

    for i in range(50):

        STATE['GLOBAL_ITERATION'] = i

        fitnesses = calculate_fitnesses(population, gt, fitness_function)

        while len(additional_population) < len(population) * population_multplier:
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
            child2.update_nodes()

            additional_population += [child1, child2]

        print("Mutating...")

        #mutations
        additional_population += mutate_population(additional_population, tensors, paths)

        population, fitnesses = join_populations(population, fitnesses, additional_population, gt, fitness_function)
        additional_population = []

        # At this place we remove duplicates and regenerate the population
        population_codes = [tree.__repr__() for tree in population]
        unique_codes = first_uniques(population_codes)

        print("Removing duplicates...")
        print(f"There are {len(population_codes) - sum(unique_codes)} duplicates in population")    

        population = [tree for tree, unique in zip(population, unique_codes) if unique]
        fitnesses = fitnesses[unique_codes]

        print("There is {} unique trees in population".format(len(population)))

        population, fitnesses = choose_pareto_rest_sorted(population, fitnesses, population_size)

        print("Best fitness:")
        print(np.max(fitnesses))
        
        if DEBUG:
            dot = draw_tree(population[np.argmax(fitnesses)])
            dot.render(f"trees/best_{STATE['GLOBAL_ITERATION']}", format='png')

    print(fitnesses)
    population[0].save_tree_architecture("./best_tree.tree")
