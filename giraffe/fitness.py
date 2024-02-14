from giraffe.tree import Tree
from giraffe.globals import BACKEND as B

from tinygrad.tensor import Tensor

import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

from enum import Enum, member


def f1_score_fitness(tree: Tree, gt: Tensor, threshold=0.5):
    pred = tree.evaluation
    pred = B.to_float((pred > threshold))
    gt = B.to_float(gt)
    return f1_score(B.to_numpy(gt).ravel(), B.to_numpy(pred).ravel()) 


def average_precision_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = B.to_float(gt)
    return average_precision_score(B.to_numpy(gt).ravel(), B.to_numpy(pred).ravel())


def binary_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):

    y_pred = B.clip(y_pred, epsilon, 1 - epsilon)
    term_0 = B.log((1 - y_true) * (1 - y_pred + epsilon))
    term_1 = B.log(y_true * (y_pred + epsilon))
    return B.mean(-(term_0 + term_1), axis=0)

def binary_cross_entropy_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = B.to_float(gt)
    return B.to_numpy(-binary_cross_entropy_loss(gt, pred)).item()


def find_distance_optimal_threshold(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = B.to_float(gt)

    precision, recall, thresholds = precision_recall_curve(B.to_numpy(gt), B.to_numpy(pred))
    fscores = (2 * precision * recall) / (precision + recall)

    ix = np.argmax(fscores)
    return thresholds[ix]


def distance_optimal_f1_score_fitness(tree: Tree, gt: Tensor):
    threshold = find_distance_optimal_threshold(tree, gt)
    return f1_score_fitness(tree, gt, threshold)


def calculate_fitnesses(population, gt, fitness_function):
    return np.array([fitness_function(tree, gt) for tree in population])


class FitnessFunction(Enum):
    F1_SCORE = member(f1_score_fitness)
    BINARY_CROSS_ENTROPY = member(binary_cross_entropy_fitness)
    AVERAGE_PRECISION = member(average_precision_fitness)
    DISTANCE_OPTIMAL_F1_SCORE = member(distance_optimal_f1_score_fitness)

    def __str__(self):
        return self.name