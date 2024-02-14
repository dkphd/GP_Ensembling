import os
from giraffe.backend.backend import Backend

def get_device():
    # check if on mac
    if os.uname().sysname == 'Darwin':
        return 'cpu' # cpu used for mac as tinygrad has problems with METAL and sometimes fails.
    else:
        return None


VERBOSE = int(os.environ.get('VERBOSE', 0))
DEVICE = os.environ.get('DEVICE', get_device())
KEEP_GRADIENTS = int(os.environ.get('KEEP_GRADIENTS', 0))

Backend.set_backend(os.environ.get('BACKEND', 'torch'))
BACKEND = Backend.get_backend()

def set_backend(backend_name):
    Backend.set_backend(backend_name)

def get_backend():
    return Backend.get_backend()
    

STATE = {
    "GLOBAL_ITERATION": 0
}