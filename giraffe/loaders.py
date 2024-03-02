from pathlib import Path

import numpy as np
from giraffe.globals import DEVICE
from giraffe.globals import BACKEND as B


def load_torch_preds_from_directory(input_path: Path):
    paths = [path.name for path in input_path.glob("*.pt")]
    tensors = [B.load_torch(input_path / path, DEVICE) for path in paths]
    return np.array(paths), tensors
