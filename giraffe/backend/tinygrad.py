from tinygrad.tensor import Tensor
from giraffe.backend.backend_interface import BackendInterface


class TinyGradBackend(BackendInterface):
    @staticmethod
    def concat(tensors, axis=0):
        return Tensor.stack(tensors, dim=axis)

    @staticmethod
    def mean(x, axis=None):
        return Tensor.mean(x, axis=axis)

    @staticmethod
    def max(x, axis=None):
        return Tensor.max(x, axis=axis)

    @staticmethod
    def min(x, axis=None):
        return Tensor.min(x, axis=axis)

    @staticmethod
    def to_numpy(x):
        return x.numpy()

    @staticmethod
    def clip(x, min, max):
        return x.clip(min, max)

    @staticmethod
    def log(x):
        return x.log()

    @staticmethod
    def to_float(x):
        return x.float()

    @staticmethod
    def load_torch(path, device="cpu"):
        import torch

        tensor = torch.load(path).numpy()
        return Tensor(tensor, device=device)

    @staticmethod
    def load_numpy(path, device="cpu"):
        import numpy as np

        return Tensor(np.load(path), device=device)

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def squeeze(x):
        return x.squeeze()
