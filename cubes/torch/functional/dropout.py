import math
import random

from torch.autograd import Function
import numpy as np
import torch
import torch.nn as nn

import cubes


class GradedDropoutFunction(Function):

    def __init__(self, a, b, tied=False):
        super().__init__()
        self.a = a
        self.b = b
        self.curr_u = None
        self.tied = tied
        self.cube = cubes.load("graded_dropout.cu")

    def forward(self, x):
        if self.curr_u is None or not self.tied:
            self.curr_u = min(random.randint(self.a, self.b), x.size(1))
        old_view = x.size()
        x = x.view(x.size(0), x.size(1), -1)
        u = self.curr_u
        grid = (x.size(0), math.ceil(x.size(2) / 64))
        block = (1, 64, 1)
        self.cube.graded_dropout_fwd_bwd(*cubes.wrap(x), self.a, self.b, u, *x.size(),
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return x.view(*old_view)

    def reset(self):
        self.curr_u = None

    def backward(self, grad):
        u = self.curr_u
        grid = (grad.size(0), math.ceil(grad.size(2) / 64))
        block = (1, 64, 1)
        grad = grad.clone() # needs fresh data pointer
        old_view = grad.size()
        grad = grad.view(grad.size(0), grad.size(1), -1)
        self.cube.graded_dropout_fwd_bwd(*cubes.wrap(grad), self.a, self.b, u, *grad.size(),
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return grad.view(*old_view)


class GradedDropoutModule(nn.Module):

    def __init__(self, a, b, tied=False):
        super().__init__()
        self.a = a
        self.b = b
        self.fn = GradedDropoutFunction(a, b, tied=tied)

    def reset(self):
        self.fn.reset()

    def forward(self, x):
        return self.fn(x)


def graded_dropout(x, a=0, b=None, training=False):
    if not training:
        return x
    if b is None:
        b = x.size(1)
    if x.is_cuda:
        fn = GradedDropoutFunction(a, b)
        return fn(x)
    else:
        raise ValueError("Non-CUDA unsupported for now")
