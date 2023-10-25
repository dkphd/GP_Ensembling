from tinygrad.tensor import Tensor

from functools import partial

class Node:
    def __init__(self, parent=None, children=[], tree=None):
        self.parent = parent
        self.children = children  # child nodes
        self.tree = tree

    def add_child(self, child_node):
        self.children.append(child_node)


class OperatorNode(Node):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.parent = parent
        self.children = children

    def calculate(self):
        pass

    def __str__(self):
        return f"OperatorNode: {None}"


class ReductionOperatorNode(OperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = lambda x: None

    def calculate(self):
        parent_eval = self.parent.evaluation if self.parent.evaluation is not None else self.parent.value

        concat = Tensor.stack([parent_eval] + [child.calculate() for child in self.children], dim=0)
        return self.operator(concat)
    

class MeanNode(ReductionOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.mean, axis=0)

    def __str__(self):
        return f"MeanNode"
    

class WeightedMeanNode(MeanNode):
    def __init__(self, parent, children, tree, weights: Tensor):
        super().__init__(parent, children, tree)

        assert len(weights.shape) == 2
        assert weights.shape[0] == 1
        assert weights.shape[1] == len(children) + 1

        self.weights = weights
        self.operator = lambda x: super().operator(x * self.weights)


class MaxNode(ReductionOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.max, axis=0)

    def __str__(self):
        return f"MaxNode"
    

class MinNode(ReductionOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.min, axis=0)

    def __str__(self):
        return f"MinNode"


class ValueNode(Node):
    def __init__(self, parent, children, tree, value):
        super().__init__(parent, children, tree)
        self.value = value
        self.evaluation = None

    def calculate(self):
        if self.children:
            self.evaluation = self.children[0].calculate()
        else:
            self.evaluation = self.value
        return self.evaluation

    def __str__(self):
        return f"ValueNode: {self.value}"

    def add_child(self, child_node):
        super().add_child(child_node)
        self.evaluation = None
