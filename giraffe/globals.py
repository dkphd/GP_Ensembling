import os

def get_device():
    # check if on mac
    if os.uname().sysname == 'Darwin':
        return 'cpu' # cpu used for mac as tinygrad has problems with METAL and sometimes fails.
    else:
        return None


def get_backend(backend=None):
    if backend == 'tinygrad' or backend is None:
        import giraffe.backend.tinygrad as backend
        return backend
    elif backend == 'torch':
        import giraffe.backend.pytorch as backend
        return backend
    else:
        raise ValueError(f"Backend {backend} not supported")


VERBOSE = int(os.environ.get('VERBOSE', 0))
DEVICE = os.environ.get('DEVICE', get_device())
KEEP_GRADIENTS = int(os.environ.get('KEEP_GRADIENTS', 0))
BACKEND = get_backend(os.environ.get('BACKEND', 'tinygrad'))

STATE = {
    "GLOBAL_ITERATION": 0
}