from pathlib import Path

import numpy as np
from tinygrad.tensor import Tensor


def load_torch_preds_from_directory(input_path: Path):

    from torch import load

    paths = [path.name for path in input_path.glob("*.pt")]
    tensors = [Tensor(load(input_path / path).numpy()) for path in paths]
    return np.array(paths), tensors