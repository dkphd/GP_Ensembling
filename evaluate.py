from argparse import ArgumentParser
from src.tree import Tree

from pathlib import Path

from torch import load

from sklearn.metrics import f1_score

def load_args():

    parser = ArgumentParser()
    parser.add_argument('-t', '--tree_path', default="./best_tree.tree")
    parser.add_argument('-p', '--predictions_path', default="./test_probs")
    parser.add_argument('-gt', '--ground_truth_path', default="./test_y.pt")

    args = parser.parse_args()

    return Path(args.tree_path), Path(args.predictions_path), Path(args.ground_truth_path)



if __name__ == "__main__":

    tree_path, predictions_path, ground_truth_path = load_args()

    tree = Tree.load_tree(tree_path, predictions_path)
    gt = load(ground_truth_path).numpy()

    tree_preds = tree.evaluation.numpy()

    f1 = f1_score(gt, tree_preds >= 0.5)

    print(f1)

