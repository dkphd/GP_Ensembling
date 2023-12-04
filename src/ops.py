import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

from src.tree import Tree
from src.node import *

from src.globals import DEBUG, STATE
from src.fitness import calculate_fitnesses



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
        plt.scatter(df['size'][~mask], df.fitness[~mask], c="b", label="non-pareto-optimal")
        plt.scatter(df['size'][mask], df.fitness[mask], c="r", label="pareto-optimal")
        plt.legend()
        plt.ylabel("fitness")
        plt.xlabel("number of nodes")
        plt.savefig(f'./pareto_plots/pareto_{STATE["GLOBAL_ITERATION"]}.png')

    population = [population[idx] for idx in np.where(mask)[0]]
    fitnesses = fitnesses[mask]
    return population, fitnesses


def choose_sorted(population, fitnesses, n):

    sizes = np.array([tree.nodes_count for tree in population], dtype=int)

    # Negate 'fitnesses' for descending order sorting
    fitnesses_neg = -np.array(fitnesses, dtype=float)

    # Create a structured array combining 'fitnesses_neg' and 'sizes'
    combined = np.array(list(zip(fitnesses_neg, sizes)), dtype=[('fitnesses_neg', float), ('sizes', int)])

    # Argsort the structured array based on 'fitnesses_neg' and 'sizes'
    argsorted = np.argsort(combined, order=['fitnesses_neg', 'sizes'])

    fitnesses = fitnesses[argsorted[:n]]

    population = [population[idx] for idx in argsorted[:n]]
    return population, fitnesses


def choose_pareto_rest_sorted(population, fitnesses, n):
    population_pareto, fitnesses_pareto = choose_pareto_optimal(population, fitnesses)
    pareto_codes = [tree.__repr__() for tree in population]

    population_sorted, fitnesses_sorted = choose_sorted(population, fitnesses, n)
    sorted_codes = [tree.__repr__() for tree in population]

    include_pareto_models = []
    include_pareto_fitnesses = []

    for code, tree, fitness in zip(pareto_codes, population_pareto, fitnesses_pareto):
        if code not in sorted_codes:
            include_pareto_models.append(tree)
            include_pareto_fitnesses.append(fitness)
    
    population = (include_pareto_models + population_sorted)[:n]
    fitnesses = np.concatenate([np.array(include_pareto_fitnesses), fitnesses_sorted])[:n]

    return population, fitnesses


def join_populations(population, fitnesses, additional_population, gt, fitness_function):

    population += additional_population
    additional_fitnesses = calculate_fitnesses(additional_population, gt, fitness_function)
    fitnesses = np.concatenate([fitnesses, additional_fitnesses])

    return population, fitnesses