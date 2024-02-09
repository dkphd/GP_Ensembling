from giraffe.tree import Tree
from tinygrad.tensor import Tensor

import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

from enum import Enum, member



def f1_score_fitness(tree: Tree, gt: Tensor, threshold=0.5):
    pred = tree.evaluation
    pred = (pred > threshold).float()
    gt = gt.float()
    return f1_score(gt.numpy(), pred.numpy()) 


def average_precision_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = gt.float()
    return average_precision_score(gt.numpy(), pred.numpy())


def binary_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):

    y_pred = y_pred.clip(epsilon, 1 - epsilon)
    term_0 = (1 - y_true) * (1 - y_pred + epsilon).log()
    term_1 = y_true * (y_pred + epsilon).log()
    return -(term_0 + term_1).mean(axis=0)

def binary_cross_entropy_fitness(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = gt.float()
    return -binary_cross_entropy_loss(gt, pred).numpy().item()



class FitnessFunction(Enum):
    F1_SCORE = member(f1_score_fitness)
    BINARY_CROSS_ENTROPY = member(binary_cross_entropy_fitness)
    AVERAGE_PRECISION = member(average_precision_fitness)

    def __str__(self):
        return self.name


def find_distance_optimal_threshold(tree: Tree, gt: Tensor):
    pred = tree.evaluation
    gt = gt.float()

    precision, recall, thresholds = precision_recall_curve(gt.numpy(), pred.numpy())
    fscores = (2 * precision * recall) / (precision + recall)

    ix = np.argmax(fscores)
    return thresholds[ix]


def distance_optimal_f1_score_fitness(tree: Tree, gt: Tensor):
    threshold = find_distance_optimal_threshold(tree, gt)
    return f1_score_fitness(tree, gt, threshold)


def calculate_fitnesses(population, gt, fitness_function):
    return np.array([fitness_function(tree, gt) for tree in population])