from pathlib import Path
from gp import main
from src.fitness import FitnessFunction, f1_score_fitness, binary_cross_entropy_fitness
from functools import partial
from gems.dask_utils.local_runner import LocalRunner
from gems.io import Json
from dask.distributed import wait
import os
from tqdm import tqdm
import pandas as pd
from torch import load


from src.tree import Tree

from pathlib import Path

from torch import load

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
import numpy as np

with open(Path('./seeds.txt'), 'r') as f:
    seeds = list(map(int, f.read().splitlines()))



default_params = {
    "input_path": "./additional_valid_probs",
    "gt_path": "./to_evolution/additional_valid_y.pt",
    "population_size": 20,
    "population_multiplier": 1,
    "tournament_size": 5,
    "fitness_function": "F1_SCORE",
    "seed": 46369,
    "tree_out_path": "./trees_benchmark_with_yalin/best_tree.tree",
}



input_paths = ["./whole_train_probs", "./additional_valid_probs"]
gt_paths = ["./to_evolution/whole_train_y.pt", "./to_evolution/additional_valid_y.pt"]
fitness_functions = ["AVERAGE_PRECISION", "F1_SCORE", "BINARY_CROSS_ENTROPY"]
allow_all_ops_options = [True, False]

# generate all combinations of parameters

base_configs = []

for input_path, gt_path in zip(input_paths, gt_paths):
    for fitness_function in fitness_functions:
        for alllow_all_ops in allow_all_ops_options:
            base_config = default_params.copy()
            base_config["input_path"] = input_path
            base_config["gt_path"] = gt_path
            base_config["fitness_function"] = fitness_function
            base_config["allow_all_ops"] = alllow_all_ops
            base_config["tree_out_path"] = f"./trees_benchmark_with_yalin/{fitness_function}_{alllow_all_ops}_{Path(input_path).name}"
            base_configs.append(base_config)
       

def make_params_from_dict(params_dict):
    params_dict['input_path'] = Path(params_dict['input_path'])
    params_dict['gt_path'] = Path(params_dict['gt_path'])
    params_dict['tree_out_path'] = Path(params_dict['tree_out_path'])
    params_dict['fitness_function'] = FitnessFunction[params_dict['fitness_function']].value

    return params_dict


for index, base_config in enumerate(base_configs):
    print(f"Running {index} out of {len(base_configs)}")
    Path(base_config["tree_out_path"]).mkdir(parents=True, exist_ok=True)
    futures = []
    for seed in tqdm(seeds[:100]):
        config = base_config.copy()
        config["seed"] = seed
        config["tree_out_path"] = f"{base_config['tree_out_path']}/{seed}.tree"
        config = make_params_from_dict(config)
        main(**config)