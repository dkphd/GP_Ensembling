from tinygrad.tensor import Tensor

from functools import partial
from typing import List, Optional, Callable, Self
from abc import ABC, abstractmethod


class Node(ABC):
    """
    Abstract Base Class for a Node in a computational tree.

    Nodes act as the fundamental building blocks of a tree,
    capable of holding children and a reference to their parent node.
    """

    def __init__(self, parent: Optional[Self] = None, children: Optional[List[Self]] = None, tree=None):
        self.parent = parent
        self.children = children if children is not None else []
        self.tree = tree

    def add_child(self, child_node: Self):
        """
        Add a child to the Node.

        Parameters:
        - child_node: Node to be added as child
        """
        self.children.append(child_node)

    def get_nodes(self):
        """
        Get all nodes in the tree created by node and it's subnodes.

        Returns:
        - List of all nodes in the tree
        """
        nodes = []
        for child in self.children:
            nodes += child.get_nodes()
        nodes.append(self)
        return nodes


class OperatorNode(Node, ABC):
    """
    Abstract Base Class for an Operator Node in a computational tree.

    Operator Nodes are specialized nodes capable of performing operations on tensors.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree):
        super().__init__(parent, children, tree)

    @abstractmethod
    def calculate(self) -> Tensor:
        """
        Abstract method for calculation logic.

        Returns:
        - Calculated Tensor object
        """
        pass


class ReductionOperatorNode(OperatorNode, ABC):
    """
    Abstract Base Class for a Reduction Operator Node in a computational tree.

    Reduction Operator Nodes are specialized Operator Nodes capable
    of performing reduction operations like mean, max, min, etc., on tensors.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree):
        super().__init__(parent, children, tree)
        self.operator: Callable[[Tensor], Tensor] = lambda x: None

    def calculate(self) -> Tensor:
        parent_eval = self.parent.evaluation if self.parent.evaluation is not None else self.parent.value
        concat = Tensor.stack([parent_eval] + [child.calculate() for child in self.children], dim=0)
        return self.operator(concat)


class MeanNode(ReductionOperatorNode):
    """
    Represents a Mean Node in a computational tree.

    A Mean Node computes the mean along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.mean, axis=0)

    def __str__(self) -> str:
        return f"MeanNode"


class WeightedMeanNode(MeanNode):
    """
    Represents a Weighted Mean Node in a computational tree.

    A Weighted Mean Node computes the mean of a tensor,
    but with different weights applied to each element.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree, weights: Tensor):
        super().__init__(parent, children, tree)

        assert len(weights.shape) == 2
        assert weights.shape[0] == 1
        assert weights.shape[1] == len(children) + 1

        self.weights = weights
        self.operator = lambda x: super().operator(x * self.weights)


class MaxNode(ReductionOperatorNode):
    """
    Represents a Max Node in a computational tree.

    A Max Node computes the maximum value along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.max, axis=0)

    def __str__(self) -> str:
        return f"MaxNode"


class MinNode(ReductionOperatorNode):
    """
    Represents a Min Node in a computational tree.

    A Min Node computes the minimum value along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree):
        super().__init__(parent, children, tree)
        self.operator = partial(Tensor.min, axis=0)

    def __str__(self) -> str:
        return f"MinNode"


class ValueNode(Node):
    """
    Represents a Value Node in a computational tree.

    A Value Node holds a specific value or tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], tree, value: Tensor):
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
        return f"ValueNode with value: {self.value} and evaluation: {self.evaluation}"

    def add_child(self, child_node):
        super().add_child(child_node)
        self.evaluation = None
