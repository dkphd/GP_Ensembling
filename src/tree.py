from tinygrad.tensor import Tensor

import numpy as np

import random

from functools import partial

from src.node import ValueNode


class Tree:
    def __init__(self, models=None):
        if models is None:
            self.models = [Tensor.randn(1, 2) for _ in range(10)]
            self.debug = True
        else:
            self.models = models
            self.debug = False

        self.root = ValueNode(None, [], self, np.random.choice(self.models))
        self.nodes = {"value_nodes": [self.root], "op_nodes": []}

    @property
    def evaluation(self):
        return self.root.evaluation if self.root.evaluation is not None else self.root.calculate()

    @property
    def nodes_count(self):
        return len(self.nodes["value_nodes"]) + len(self.nodes["op_nodes"])
