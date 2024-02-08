from src.tree import Tree
import torch
import tempfile
from pathlib import Path

class InferenceTree:
    """
    Can be used to directly perform inference with a tree from input data.
    It is generally not recommended to use it as it is not optimized for performance and may be unstable.
    """

    def __init__(self, tree_architecture_path, prediction_functions: dict):
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

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            for model_id, prediction_function in self.prediction_functions.items():
                predicted = prediction_function(input_data)
                if not isinstance(predicted, torch.Tensor):
                    try:
                        predicted = torch.tensor(predicted)
                    except ValueError:
                        raise Exception("Prediction function did not return a tensor or something that can be converted to a tensor")
                torch.save(predicted, temp_dir / model_id)
            
            tree, _ = Tree.load_tree(self.tree_architecture_path, temp_dir)
            return tree.evaluation.numpy()