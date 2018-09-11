import functools
import os

from collections import namedtuple
from cupy.cuda import function
from pynvrtc.compiler import Program


Stream = namedtuple("Stream", "ptr")


class Cube(object):

    class CubeFunction(object):

        def __init__(self, function):
            self.function = function

        def __call__(self, *args, **kwargs):
            kwargs["stream"] = Stream(kwargs["stream"])
            self.function(args=args, **kwargs)

    def __init__(self, src_module):
        self.src_module = src_module

    def __getattr__(self, name):
        return self.CubeFunction(self.src_module.get_function(name))


@functools.lru_cache(maxsize=1024)
def load(filename):
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cuda", filename))
    with open(filename) as f:
        return loads(f.read(), os.path.basename(filename))


def loads(kernel_str, kernel_name):
    program = Program(kernel_str, kernel_name)
    ptx = program.compile()
    src_module = function.Module()
    src_module.load(bytes(ptx.encode()))
    return Cube(src_module)


def wrap(*args, origin="pytorch"):
    if origin == "pytorch":
        return [x.data_ptr() for x in args]
    else:
        raise ValueError("Origin must be PyTorch.")
