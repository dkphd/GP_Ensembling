import torch
from giraffe.backend.backend_interface import BackendInterface

class PyTorchBackend(BackendInterface):

    # rewrite all not call the already defined functions

    @staticmethod
    def concat(tensors, axis=0):
        # check if tensors are not unidimensional, if so we need to add singular dimension before concatenating
        print([t.shape for t in tensors])
        if len(tensors[0].shape) == 1:
            tensors = [t.unsqueeze(0) for t in tensors]
        return torch.cat(tensors, dim=axis)
    
    @staticmethod
    def mean(x, axis=None):
        return torch.mean(x, axis=axis)
    
    @staticmethod
    def max(x, axis=None):
        return torch.max(x, axis=axis)
    
    @staticmethod
    def min(x, axis=None):
        return torch.min(x, axis=axis)
    
    @staticmethod
    def to_numpy(x):
        return x.detach().numpy()

    @staticmethod
    def clip(x, min, max):
        return torch.clamp(x, min, max)

    @staticmethod
    def log(x):
        return torch.log(x)

    @staticmethod
    def to_float(x):
        return x.float()

    @staticmethod
    def load_torch(path, device="cpu"):
        return torch.load(path, map_location=device)