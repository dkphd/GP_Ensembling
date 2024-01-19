import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

from src.tree import Tree
from src.node import *

from src.globals import VERBOSE, STATE
from src.fitness import calculate_fitnesses

from pathlib import Path
import matplotlib as mpl

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


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
    df['mask'] = mask
    if VERBOSE:
        fig, ax = plt.subplots(figsize=(8, 4))

        # Load your custom font
        fpath = Path(mpl.get_data_path(), "fonts/ttf/times.ttf")
        prop = FontProperties(fname=fpath)

        # Apply the font globally
        mpl.rcParams['font.family'] = prop.get_name()

        ax.scatter(df['size'][~mask], df.fitness[~mask], c="black")
        ax.scatter(df['size'][mask], df.fitness[mask], c="#62b879")
        ax.set_xlabel("Times New Roman")
        # plt.legend()

        fig.savefig(f'./pareto_plots/pareto_{STATE["GLOBAL_ITERATION"]}.png')
        df.to_csv(f'./pareto_plots/pareto_{STATE["GLOBAL_ITERATION"]}.csv', index=False)

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
    pareto_codes = [tree.__repr__() for tree in population_pareto]

    population_sorted, fitnesses_sorted = choose_sorted(population, fitnesses, n)
    sorted_codes = [tree.__repr__() for tree in population_sorted]

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