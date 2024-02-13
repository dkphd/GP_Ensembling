from tinygrad.tensor import Tensor

def concat(tensors, axis=0):
    return Tensor.stack(tensors, dim=axis)

def mean(x, axis=None):
    return Tensor.mean(x, axis=axis)

def max(x, axis=None):
    return Tensor.max(x, axis=axis)

def min(x, axis=None):
    return Tensor.min(x, axis=axis)

def to_numpy(x):
    return x.numpy()

def clip(x, min, max):
    return x.clip(min, max)

def log(x):
    return x.log()

def to_float(x):
    return x.float()

def load_torch(path, device="cpu"):
    import torch
    with torch.no_grad():
        tensor = torch.load(path).numpy()
        return Tensor(tensor, device=device)
    