from src.node import OperatorNode

from functools import reduce
from operator import add, sub, mul, truediv


class BinaryOperatorNode(OperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = lambda x: None

    def calculate(self):
        return reduce(
            lambda x, y: self.operator(x, y),
            [self.parent.evaluation if self.parent.evaluation is not None else self.parent.value]
            + [child.calculate() for child in self.children],
        )

    def __str__(self):
        return f"BinaryOperatorNode: {self.operator}"


class AddNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = add


class SubNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = sub


class MulNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = mul


class DivNode(BinaryOperatorNode):
    def __init__(self, parent, children, tree):
        super().__init__(parent, children, tree)
        self.operator = truediv
