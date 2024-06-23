from giraffe.backend.backend_interface import BackendInterface


class Backend:
    _current_backend: BackendInterface = None

    @classmethod
    def set_backend(cls, backend_name):  # TODO: Add option to set backend by providing class instead
        if backend_name == "torch":
            from giraffe.backend.pytorch import PyTorchBackend

            cls._current_backend = PyTorchBackend
        elif backend_name == "tinygrad":
            from giraffe.backend.tinygrad import TinyGradBackend

            cls._current_backend = TinyGradBackend
        else:
            raise ValueError(f"Invalid backend: {backend_name}")

    @classmethod
    def get_backend(cls):
        if cls._current_backend is None:
            cls.set_backend("torch")  # Set a default
        return cls._current_backend
