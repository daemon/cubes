import math
import random

from torch.autograd import Function
import numpy as np
import torch
import torch.nn as nn

import cubes


class GradedDropoutFunction(Function):

    @staticmethod
    def forward(ctx, x, options):
        ctx.options = options
        if options.get("u") is None or not options["tied"]:
            options["u"] = min(random.randint(options["a"], options["b"]), x.size(1))
        if not options["inplace"]:
            x = x.clone().detach()
        u = options["u"]
        bsz, csz, hsz = x.size(0), x.size(1), np.prod(x.size()[2:])
        grid = (x.size(0), math.ceil(hsz / 64))
        block = (1, 64, 1)
        cube = cubes.load("graded_dropout.cu")
        cube.graded_dropout_fwd_bwd(*cubes.wrap(x), options["a"], options["b"], u, bsz, csz, hsz,
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        return x

    @staticmethod
    def backward(ctx, grad):
        options = ctx.options
        u = options["u"]
        bsz, csz, hsz = grad.size(0), grad.size(1), np.prod(grad.size()[2:])
        grid = (grad.size(0), math.ceil(hsz / 64))
        block = (1, 64, 1)
        grad = grad.clone().detach() # needs fresh data pointer
        cube = cubes.load("graded_dropout.cu")
        cube.graded_dropout_fwd_bwd(*cubes.wrap(grad), options["a"], options["b"], u, bsz, csz, hsz,
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        return grad, None


class GradedDropoutModule(nn.Module):

    def __init__(self, a, b, tied=False, inplace=False, eval_u=None):
        super().__init__()
        self.a = a
        self.b = b
        self.eval_u = eval_u
        self.options = dict(u=None, tied=tied, a=a, b=b, inplace=inplace)

    def reset(self):
        self.options["u"] = None

    def forward(self, x):
        if not self.training and self.eval_u is None:
            return x
        if self.eval_u is not None and not self.training:
            self.options["u"] = self.eval_u
            self.options["tied"] = True
        return GradedDropoutFunction.apply(x, self.options)


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
