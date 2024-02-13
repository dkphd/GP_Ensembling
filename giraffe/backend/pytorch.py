import torch

def concat(tensors, axis=0):

    # check if tensors are not unidimensional, if so we need to add singular dimension before concatenating
    if len(tensors[0].shape) == 1:
        tensors = [t.unsqueeze(0) for t in tensors]
        
    return torch.cat(tensors, dim=axis)

def mean(x, axis=None):
    return torch.mean(x, axis=axis)

def max(x, axis=None):
    return torch.max(x, axis=axis)

def min(x, axis=None):
    return torch.min(x, axis=axis)

def to_numpy(x):
    return x.detach().numpy()

def clip(x, min, max):
    return torch.clamp(x, min, max)

def log(x):
    return torch.log(x)

def to_float(x):
    return x.float()

def load_torch(path, device="cpu"):
    return torch.load(path, map_location=device)