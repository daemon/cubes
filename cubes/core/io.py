import os

from pycuda.compiler import SourceModule
from pycuda.driver import PointerHolderBase
import pycuda.driver as drv
import pycuda.autoinit


class PointerWrapper(PointerHolderBase):

    def __init__(self, pointer):
        super().__init__()
        self.gpudata = pointer
        self.pointer = pointer

    def get_pointer(self):
        return self.pointer


class Cube(object):

    class CubeFunction(object):

        def __init__(self, function):
            self.function = function

        def __call__(self, *args, **kwargs):
            self.function(*args, **kwargs)

    def __init__(self, src_module):
        self.src_module = src_module

    def __getattr__(self, name):
        return self.CubeFunction(self.src_module.get_function(name))


def load(filename):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuda", filename)
    with open(filename) as f:
        return loads(f.read())


def loads(kernel_str):
    return Cube(SourceModule(kernel_str))


def wrap(*args, origin="pytorch"):
    if origin == "pytorch":
        return [PointerWrapper(x.data_ptr()) for x in args]
    else:
        raise ValueError("Origin must be PyTorch.")
