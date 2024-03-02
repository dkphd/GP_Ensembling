import torch
import tempfile
from pathlib import Path
from giraffe.globals import BACKEND as B
from giraffe.backend.backend import Backend
from giraffe.tree import Tree


class InferenceTree:
    """
    Can be used to directly perform inference with a tree from input data.
    It is generally not recommended to use it as it is not optimized for performance and may be unstable.
    """

    def __init__(self, tree_architecture_path, prediction_functions: dict, backend="torch"):
        Backend.set_backend(backend)
        self.backend = backend

        self.tree_architecture_path = tree_architecture_path
        self.prediction_functions = prediction_functions

        self.empty_tree = Tree.load_tree_architecture(tree_architecture_path)

        for model_id in self.empty_tree.get_unique_value_node_ids():
            if model_id not in prediction_functions:
                raise Exception(f"Model with id {model_id} does not have a prediction function")

    def predict(self, input_data):
        """
        Perform inference with the input data.
        """

        if self.backend == "tinygrad":
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                for model_id, prediction_function in self.prediction_functions.items():
                    predicted = prediction_function(input_data)
                    predicted = self.to_torch_tensor(predicted)
                    torch.save(predicted, temp_dir / model_id)

                tree, _ = Tree.load_tree(self.tree_architecture_path, temp_dir)
                return B.to_numpy(tree.evaluation)

        elif self.backend == "torch":
            tensors = {}
            for model_id, prediction_function in self.prediction_functions.items():
                predicted = prediction_function(input_data)
                predicted = self.to_torch_tensor(predicted)
                tensors[model_id] = predicted

            tree, _ = Tree.load_tree(self.tree_architecture_path, Path("."), tensors)
            return tree.evaluation
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @staticmethod
    def to_torch_tensor(data):
        if isinstance(data, torch.Tensor):
            return data
        else:
            try:
                return torch.tensor(data)
            except Exception as e:
                raise Exception("Could not convert data to torch tensor:", e)
