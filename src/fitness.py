from src.tree import Tree
from tinygrad.tensor import Tensor

import numpy as np
from sklearn.metrics import f1_score

from enum import Enum


def f1_score_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    pred = (pred > 0.5).float()
    gt = gt.float()
    return f1_score(gt.numpy(), pred.numpy()) 


def binary_cross_entropy_loss(y_true, y_pred): 
    loss = - (y_true * y_pred.log() + (1 - y_true) * (1-y_pred).log()).mean()
    return loss


def binary_cross_entropy_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = gt.float()
    return -binary_cross_entropy_loss(gt, pred).numpy().item()



class FitnessFunction(Enum):
    F1_SCORE = f1_score_fitness
    BINARY_CROSS_ENTROPY = binary_cross_entropy_fitness

    def __str__(self):
        return self.name
    

def calculate_fitnesses(population, gt, fitness_function):
    return np.array([fitness_function(tree, gt) for tree in population])