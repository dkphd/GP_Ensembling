from abc import ABC


class BackendInterface(ABC):
    @staticmethod
    def concat(self, tensors, axis=0):
        raise NotImplementedError()

    @staticmethod
    def mean(self, x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def max(self, x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def min(self, x, axis=None):
        raise NotImplementedError()

    @staticmethod
    def to_numpy(self, x):
        raise NotImplementedError()

    @staticmethod
    def clip(self, x, min, max):
        raise NotImplementedError()

    @staticmethod
    def log(self, x):
        raise NotImplementedError()

    @staticmethod
    def to_float(self, x):
        raise NotImplementedError()
