from functools import partial
from typing import List, Optional, Callable, Self, Union
from abc import ABC, abstractmethod

from giraffe.globals import BACKEND as B
from giraffe.types import Tensor

import numpy as np


class Node(ABC):
    """
    Abstract Base Class for a Node in a computational tree.

    Nodes act as the fundamental building blocks of a tree,
    capable of holding children and a reference to their parent node.
    """

    def __init__(self, parent: Optional[Self] = None, children: Optional[List[Self]] = None):
        self.parent = parent
        self.children = children if children is not None else []
        self.type = None

    def add_child(self, child_node: Self):
        """
        Add a child to the Node.

        Parameters:
        - child_node: Node to be added as child
        """
        self.children.append(child_node)
        child_node.parent = self

    def remove_child(self, child_node: Self):
        self.children.remove(child_node)

    def replace_child(self, child, replacement_node):
        self.add_child(replacement_node)
        self.remove_child(child)

    def get_nodes(self):  # TODO: This is not topologically sorted, change that
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

    @abstractmethod
    def copy(self):
        """
        Create a copy of the node.
        It's children and parent references are not copied.

        Returns:
        - Copy of the node
        """
        pass

    def copy_subtree(self):
        """
        Copy the subtree rooted at this node.

        Returns:
        - Copy of the subtree rooted at this node
        """
        self_copy = self.copy()
        self_copy.children = [child.copy_subtree() for child in self.children]
        for child in self_copy.children:
            child.parent = self_copy
        return self_copy

    @abstractmethod
    def calculate(self) -> Tensor:
        """
        Abstract method for calculation logic.

        Returns:
        - Calculated Tensor object
        """
        pass

    @property
    @abstractmethod
    def code(self) -> str:
        """
        Identifies node for duplicate handling.

        Returns:
        - Code string
        """
        pass

    def __repr__(self):
        return self.code()


class ValueNode(Node):
    """
    Represents a Value Node in a computational tree.

    A Value Node holds a specific value or tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], value: Tensor, id: Union[int, str]):
        super().__init__(parent, children)
        self.value = value
        self.evaluation = None
        self.id = id

    def calculate(self):
        if self.children:
            for child in self.children:
                self.evaluation = child.calculate()
        else:
            self.evaluation = self.value
        return self.evaluation

    def __str__(self):
        return f"ValueNode with value at: {hex(id(self.value))}"  # and evaluation: {self.evaluation}"

    def add_child(self, child_node):
        super().add_child(child_node)
        self.evaluation = None

    def copy(self):
        return ValueNode(None, None, self.value, self.id)

    @property
    def code(self) -> str:
        return f"VN[{self.id}]"


class OperatorNode(Node, ABC):
    """
    Abstract Base Class for an Operator Node in a computational tree.

    Reduction Operator Nodes are specialized Operator Nodes capable
    of performing reduction operations like mean, max, min, etc., on tensors.
    """

    def __init__(
        self,
        parent: Optional[Node],
        children: Optional[List[Node]],
        operator: Callable[[Tensor], Tensor] = lambda x: None,
    ):
        super().__init__(parent, children)
        self.operator = operator

    def calculate(self) -> Tensor:
        parent_eval = self.parent.evaluation if self.parent.evaluation is not None else self.parent.value
        concat = B.concat(
            [B.unsqueeze(parent_eval, axis=0)] + [B.unsqueeze(child.calculate(), axis=0) for child in self.children],
            axis=0,
        )
        return self.operator(concat)

    @staticmethod
    @abstractmethod
    def create_node(parent, children):
        raise NotImplementedError()


class MeanNode(OperatorNode):
    """
    Represents a Mean Node in a computational tree.

    A Mean Node computes the mean along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]]):
        super().__init__(parent, children, MeanNode.get_operator())

    def __str__(self) -> str:
        return "MeanNode"

    def copy(self):
        return MeanNode(None, None)

    @property
    def code(self) -> str:
        return "MN"

    @staticmethod
    def get_operator():
        return partial(B.mean, axis=0)

    def adjust_params(self):
        return

    @staticmethod
    def create_node(parent, children):  # TODO: it could be derived from simple vs parametrized OperatorNode
        return MeanNode(parent, children)


class WeightedMeanNode(MeanNode):
    """
    Represents a Weighted Mean Node in a computational tree.

    A Weighted Mean Node computes the mean of a tensor,
    but with different weights applied to each element.
    """

    def __init__(
        self,
        parent: ValueNode,
        children: List[ValueNode],
        weights: List[float],
    ):
        self._weights = weights
        super().__init__(parent, children)

        self._node_weight_assertion()

        self.operator = self.op

    def op(self, x):
        weight_shape = (-1, *([1] * (len(x.shape) - 1)))
        w = B.reshape(self.weights, weight_shape)
        x = x * w
        x = B.sum(x, axis=0)
        return x

    def copy(self):
        return WeightedMeanNode(None, [], [x for x in self._weights])

    def add_child(self, child_node: ValueNode):
        assert isinstance(child_node, ValueNode)
        child_weight = np.random.uniform(0, 1)
        adj = 1.0 - child_weight

        for i, val in enumerate(self._weights):
            self._weights[i] = val * adj

        self._weights.append(child_weight)
        child_node.parent = self
        self.children.append(child_node)

        self._node_weight_assertion()
        assert len(self._weights) == (
            len(self.children) + 1
        ), "Length of weight array is different than number of adjacent nodes"

    def remove_child(self, child_node: ValueNode):
        assert isinstance(child_node, ValueNode)

        child_ix = self.children.index(child_node)

        adj = 1.0 - self._weights[child_ix]
        self._weights.pop(child_ix)
        self.children.pop(child_ix)

        for i, val in enumerate(self._weights):
            self._weights[i] = val / adj

        self._node_weight_assertion()
        assert len(self._weights) == (
            len(self.children) + 1
        ), "Length of weight array is different than number of adjacent nodes"

    def replace_child(self, child, replacement_node):
        child_ix = self.children.index(child)
        self.children[child_ix] = replacement_node
        replacement_node.parent = self
        assert len(self._weights) == (
            len(self.children) + 1
        ), "Length of weight array is different than number of adjacent nodes"

    def __str__(self) -> str:
        return f"WeightedMeanNode with weights: {B.to_numpy(B.tensor(self._weights)).round(2)}"

    @property
    def code(self) -> str:
        return "WMN"

    @staticmethod  # this should not be static because for parametrized it doesnt work well
    def get_operator():
        return partial(B.mean, axis=0)

    @property
    def weights(self):
        w = B.tensor(self._weights)
        return w

    @staticmethod
    def create_node(parent, children):
        weights = [np.random.uniform(0, 1)]
        weight_left = 1 - weights[0]
        for child in children[:-1]:
            weights.append(np.random.uniform(0, weight_left))
            weight_left -= weights[-1]
        weights.append(weight_left)

        return WeightedMeanNode(parent, children, weights)

    def _node_weight_assertion(self):
        assert np.isclose(np.sum(self._weights), 1), "Weights do not sum to 1"


class MaxNode(OperatorNode):
    """
    Represents a Max Node in a computational tree.

    A Max Node computes the maximum value along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]]):
        super().__init__(parent, children, MaxNode.get_operator())

    def __str__(self) -> str:
        return "MaxNode"

    def copy(self):
        return MaxNode(None, None)

    @property
    def code(self) -> str:
        return "MAX"

    @staticmethod
    def get_operator():
        return partial(B.max, axis=0)

    def adjust_params(self):
        return

    def remove_child(self, child_node: Self):
        assert len(self.children) > 1, "SHOULD REMOVE ONLY CHILD"
        return super().remove_child(child_node)

    @staticmethod
    def create_node(parent, children):  # TODO: it could be derived from simple vs parametrized OperatorNode
        return MaxNode(parent, children)


class MinNode(OperatorNode):
    """
    Represents a Min Node in a computational tree.

    A Min Node computes the minimum value along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]]):
        super().__init__(parent, children, MinNode.get_operator())

    def __str__(self) -> str:
        return "MinNode"

    def copy(self):
        return MinNode(None, None)

    @property
    def code(self) -> str:
        return "MIN"

    @staticmethod
    def get_operator():
        return partial(B.min, axis=0)

    def adjust_params(self):
        return

    @staticmethod
    def create_node(parent, children):  # TODO: it could be derived from simple vs parametrized OperatorNode
        return MinNode(parent, children)
