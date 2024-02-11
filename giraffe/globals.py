import os

def get_device():
    # check if on mac
    if os.uname().sysname == 'Darwin':
        return 'cpu' # cpu used for mac as tinygrad has problems with METAL and sometimes fails.
    else:
        return None


VERBOSE = int(os.environ.get('VERBOSE', 0))
DEVICE = os.environ.get('DEVICE', get_device())


STATE = {
    "GLOBAL_ITERATION": 0
}