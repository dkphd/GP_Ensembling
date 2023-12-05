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

def load_args():

    parser = ArgumentParser()
    parser.add_argument("--input_path", default="./additional_valid_probs")
    parser.add_argument("--gt_path", default="./to_evolution/additional_valid_y.pt")
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--population_multiplier", type=int, default=1)
    parser.add_argument("--tournament_size", type=int, default=5)
    parser.add_argument("--fitness_function", type=lambda func: FitnessFunction[func].value, choices=list(FitnessFunction), default="F1_SCORE")
    parser.add_argument("--allow_all_ops", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--tree_out_path", type=str, default="./best_tree.tree")
    args = parser.parse_args()

    return Path(args.input_path), Path(args.gt_path), args.population_size, args.population_multiplier, args.tournament_size, args.fitness_function, args.allow_all_ops, args.seed, args.tree_out_path


def main(input_path, gt_path, population_size, population_multiplier, tournament_size, fitness_function, allow_all_ops, seed, tree_out_path):

    np.random.seed(seed)

    paths, tensors = load_torch_preds_from_directory(input_path)

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

        while len(additional_population) < len(population) * population_multiplier:
            idxs = np.random.choice(ids, tournament_size, replace=False)
            tournament_fitnesses = np.array(fitnesses)[idxs]
            parent_idxs = idxs[np.argsort(tournament_fitnesses)[-2:]]
            parent1, parent2 = population[parent_idxs[0]], population[parent_idxs[1]]

            try:
                child1, child2 = crossover(parent1, parent2, mutation_chance_crossover=False, debug=DEBUG)
            except Exception as e:
                if DEBUG:
                    print("Crossover failed due to: ", e)
                continue

            child1.update_nodes()
            child2.update_nodes()

            additional_population += [child1, child2]


        if DEBUG > 1:
            print("Mutating...")

        #mutations
        additional_population += mutate_population(additional_population, tensors, paths, allow_all_ops=allow_all_ops)

        population, fitnesses = join_populations(population, fitnesses, additional_population, gt, fitness_function)
        additional_population = []

        # At this place we remove duplicates and regenerate the population
        population_codes = [tree.__repr__() for tree in population]
        unique_codes = first_uniques(population_codes)


        if DEBUG > 1:
            print("Removing duplicates...")
            print(f"There are {len(population_codes) - sum(unique_codes)} duplicates in population")    

        population = [tree for tree, unique in zip(population, unique_codes) if unique]
        fitnesses = fitnesses[unique_codes]

        if DEBUG > 1:
            print("There are {} unique trees in population".format(len(population)))

        population, fitnesses = choose_pareto_rest_sorted(population, fitnesses, population_size)

        if DEBUG > 1:
            print("Best fitness:")
            print(np.max(fitnesses))
        
        if DEBUG > 1:
            dot = draw_tree(population[np.argmax(fitnesses)])
            dot.render(f"trees/best_{STATE['GLOBAL_ITERATION']}", format='png')

    print(fitnesses)
    population[0].save_tree_architecture(tree_out_path)
    return population[0]


if __name__ == "__main__":
    
    input_path, gt_path, population_size, population_multiplier, tournament_size, fitness_function, allow_all_ops, seed, tree_out_path = load_args()

    main(input_path, gt_path, population_size, population_multiplier, tournament_size, fitness_function, allow_all_ops, seed, tree_out_path)
