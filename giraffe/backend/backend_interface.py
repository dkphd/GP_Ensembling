from abc import ABC

# TODO: Backend interface could be otherwise a Tensor class instead


class BackendInterface(ABC):
    @staticmethod
    def tensor(x):
        raise NotImplementedError()

    @staticmethod
    def concat(tensors, axis=0):
        raise NotImplementedError()

    @staticmethod
    def mean(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def max(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def min(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def sum(x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def to_numpy(x):
        raise NotImplementedError()

    @staticmethod
    def clip(x, min, max):
        raise NotImplementedError()

    @staticmethod
    def log(x):
        raise NotImplementedError()

    @staticmethod
    def to_float(x):
        raise NotImplementedError()

    @staticmethod
    def shape(x):
        raise NotImplementedError()

    @staticmethod
    def reshape(x, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def squeeze(x):
        raise NotImplementedError()

    @staticmethod
    def unsqueeze(x, axis):
        raise NotImplementedError()
