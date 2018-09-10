import math
import random

from torch.autograd import Function
import numpy as np
import torch

import cubes


class GradedDropoutFunction(Function):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.cube = cubes.load("graded_dropout.cu")

    def forward(self, x):
        u = min(random.randint(self.a, self.b), x.size(1))
        self.curr_u = u
        grid = (x.size(0), math.ceil(x.size(2) / 64))
        block = (1, 64, 1)
        self.cube.graded_dropout_fwd_bwd(*cubes.wrap(x), self.a, self.b, u, *x.size(),
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return x

    def backward(self, grad):
        u = self.curr_u
        grid = (grad.size(0), math.ceil(grad.size(2) / 64))
        block = (1, 64, 1)
        grad = grad.clone() # needs fresh data pointer
        self.cube.graded_dropout_fwd_bwd(*cubes.wrap(grad), self.a, self.b, u, *grad.size(),
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return grad


def main():
    fn = GradedDropoutFunction(20, 40)
    x = torch.ones(1, 40, 5).cuda()
    x.requires_grad = True
    t = fn(x).sum()
    t.backward()


if __name__ == "__main__":
    main()
