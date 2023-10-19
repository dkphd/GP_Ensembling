from src.tree import *

if __name__ == "__main__":
    tree = Tree()
    tree.grow_tree(tree.root, max_depth=3)
    dot = draw_tree(tree.root)
    dot.render("tree", view=False, format="png")

    # Calculate the tree
    print(tree.root.calculate())
    dot = draw_tree(tree.root)
    tree.root.evaluation.numpy()
    dot.render("tree2", view=False, format="png")
