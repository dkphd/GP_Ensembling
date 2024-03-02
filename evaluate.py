from argparse import ArgumentParser
from giraffe.tree import Tree

from pathlib import Path

from torch import load

from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
from giraffe.fitness import find_distance_optimal_threshold
from giraffe.globals import BACKEND as B
from tinygrad.tensor import Tensor

def load_args():

    parser = ArgumentParser()
    parser.add_argument('-t', '--tree_path', default="./best_tree_3.tree")
    parser.add_argument('-p', '--predictions_path', default="./test_probs")
    parser.add_argument('-gt', '--ground_truth_path', default="./gt/test_labels.pt")

    args = parser.parse_args()

    return Path(args.tree_path), Path(args.predictions_path), Path(args.ground_truth_path)



if __name__ == "__main__":

    tree_path, predictions_path, ground_truth_path = load_args()

    tree, _ = Tree.load_tree(tree_path, predictions_path)
    gt = B.to_numpy(load(ground_truth_path))

    tree_preds =B.to_numpy(tree.evaluation)

    f1 = f1_score(gt, tree_preds > 0.558)

    print(find_distance_optimal_threshold(tree, Tensor(gt)))

    print(f1)
    print(confusion_matrix(gt, tree_preds > 0.558))
