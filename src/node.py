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


class OperatorNode(Node, ABC):
    """
    Abstract Base Class for an Operator Node in a computational tree.

    Operator Nodes are specialized nodes capable of performing operations on tensors.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], operator: Callable[[Tensor], Tensor] = lambda x: None):
        super().__init__(parent, children)
        self.operator = operator


class ReductionOperatorNode(OperatorNode, ABC):
    """
    Abstract Base Class for a Reduction Operator Node in a computational tree.

    Reduction Operator Nodes are specialized Operator Nodes capable
    of performing reduction operations like mean, max, min, etc., on tensors.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], operator: Callable[[Tensor], Tensor] = lambda x: None):
        super().__init__(parent, children, operator)

    def calculate(self) -> Tensor:
        parent_eval = self.parent.evaluation if self.parent.evaluation is not None else self.parent.value
        concat = Tensor.stack([parent_eval] + [child.calculate() for child in self.children], dim=0)
        return self.operator(concat)


class MeanNode(ReductionOperatorNode):
    """
    Represents a Mean Node in a computational tree.

    A Mean Node computes the mean along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]]):
        super().__init__(parent, children, partial(Tensor.mean, axis=0))

    def __str__(self) -> str:
        return f"MeanNode"
    

    def copy(self):
        return MeanNode(None, None)


class WeightedMeanNode(MeanNode):
    """
    Represents a Weighted Mean Node in a computational tree.

    A Weighted Mean Node computes the mean of a tensor,
    but with different weights applied to each element.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], weights: Tensor):
        
        assert len(weights.shape) == 2
        assert weights.shape[0] == 1
        assert weights.shape[1] == len(children) + 1

        self.weights = weights

        super().__init__(parent, children, lambda x: super().operator(x * self.weights))


    def copy(self):
        return WeightedMeanNode(None, None, self.weights)

    
    def add_child(self, child_node: Self):
        raise Exception("Adding child to weighted mean node is currently not supported due to the way weights are handled.")

    def __str__(self) -> str:
        return f"WeightedMeanNode with weights: {self.weights.numpy():.2f}"


class MaxNode(ReductionOperatorNode):
    """
    Represents a Max Node in a computational tree.

    A Max Node computes the maximum value along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]]):
        super().__init__(parent, children, partial(Tensor.max, axis=0))

    def __str__(self) -> str:
        return f"MaxNode"
    
    def copy(self):
        return MaxNode(None, None)



class MinNode(ReductionOperatorNode):
    """
    Represents a Min Node in a computational tree.

    A Min Node computes the minimum value along a specified axis of a tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]]):
        super().__init__(parent, children, partial(Tensor.min, axis=0))

    def __str__(self) -> str:
        return f"MinNode"


    def copy(self):
        return MinNode(None, None)


class ValueNode(Node):
    """
    Represents a Value Node in a computational tree.

    A Value Node holds a specific value or tensor.
    """

    def __init__(self, parent: Optional[Node], children: Optional[List[Node]], value: Tensor, id: int = None):
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
        return f"ValueNode with value id: {hex(id(self.value))}"# and evaluation: {self.evaluation}"

    def add_child(self, child_node):
        super().add_child(child_node)
        self.evaluation = None

    def copy(self):
        return ValueNode(None, None, self.value, self.id)
    