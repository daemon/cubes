import math

from torch.autograd import Function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cubes


class L0WeightTransformFunction(Function):

    @staticmethod
    def forward(ctx, weights, log_alpha, beta, gamma, zeta, is_training):
        cube = cubes.load("l0_sparsity.cu")
        ctx.hyperparams = beta, gamma, zeta
        csz, gsz = weights.size(0), np.prod(weights.size()[1:])
        grid = (math.ceil(csz / 16), math.ceil(gsz / 16))
        block = (16, 16, 1)
        out_weights = weights.new(*weights.size())
        if is_training:
            uniform = log_alpha.new(*log_alpha.size()).uniform_()
            cube.l0_weights_fwd(*cubes.wrap(out_weights, weights, log_alpha, uniform), np.float32(beta),
                np.float32(gamma), np.float32(zeta), csz, gsz, grid=grid, block=block,
                stream=torch.cuda.current_stream().cuda_stream)
            ctx.save_for_backward(weights, log_alpha, uniform)
        else:
            cube.l0_weights_test_fwd(*cubes.wrap(out_weights, weights, log_alpha), np.float32(beta),
                np.float32(gamma), np.float32(zeta), csz, gsz, grid=grid, block=block,
                stream=torch.cuda.current_stream().cuda_stream)
        return out_weights

    @staticmethod
    def backward(ctx, weights_grad):
        cube = cubes.load("l0_sparsity.cu")
        weights, log_alpha, uniform = ctx.saved_variables
        beta, gamma, zeta = ctx.hyperparams
        csz, gsz = weights_grad.size(0), np.prod(weights_grad.size()[1:])
        grid = (math.ceil(csz / 16), math.ceil(gsz / 16))
        block = (16, 16, 1)
        weights_grad = weights_grad.clone().detach() # needs fresh data pointer
        out_weights_grad = weights_grad.new(*weights_grad.size())
        out_log_alpha_grad = weights_grad.new(*weights_grad.size())
        cube.l0_weights_bwd(*cubes.wrap(out_weights_grad, out_log_alpha_grad, weights_grad, weights, 
            log_alpha, uniform), np.float32(beta), np.float32(gamma), np.float32(zeta), csz, gsz,
            grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        out_log_alpha_grad = out_log_alpha_grad.view(log_alpha.size(0), -1)
        return out_weights_grad, out_log_alpha_grad.sum(1), None, None, None, None


class L0NormFunction(Function):

    @staticmethod
    def forward(ctx, weights, log_alpha, beta, gamma, zeta):
        cube = cubes.load("l0_sparsity.cu")
        csz, gsz = weights.size(0), np.prod(weights.size()[1:])
        ctx.hyperparams = beta, gamma, zeta, csz, gsz
        grid = (math.ceil(csz / 64), 1)
        block = (64, 1, 1)
        out_norm = log_alpha.new(*log_alpha.size())
        cube.l0_norm_fwd(*cubes.wrap(out_norm, log_alpha), np.float32(beta), np.float32(gamma), 
            np.float32(zeta), csz, gsz, grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        ctx.save_for_backward(log_alpha)
        return out_norm

    @staticmethod
    def backward(ctx, norm_grad):
        cube = cubes.load("l0_sparsity.cu")
        beta, gamma, zeta, csz, gsz = ctx.hyperparams
        grid = (math.ceil(csz / 64), 1)
        block = (64, 1, 1)
        log_alpha, = ctx.saved_variables
        norm_grad = norm_grad.clone().detach() # needs fresh data pointer
        out_norm_grad = log_alpha.new(*log_alpha.size())
        cube.l0_norm_bwd(*cubes.wrap(out_norm_grad, norm_grad, log_alpha), np.float32(beta), np.float32(gamma),
            np.float32(zeta), csz, gsz, grid=grid, block=block, stream=torch.cuda.current_stream().cuda_stream)
        return None, out_norm_grad, None, None, None


l0_weight_transform = L0WeightTransformFunction.apply
l0_norm = L0NormFunction.apply


class L0Linear(nn.Module):

    def __init__(self, *args, beta=2/3, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.hyperparams = beta, gamma, zeta
        self.linear = nn.Linear(*args)
        self.log_alpha = nn.Parameter(torch.empty(self.linear.weight.size(0)).normal_(0, 0.01))

    @property
    def l0_norm(self):
        return l0_norm(self.linear.weight, self.log_alpha, *self.hyperparams).sum() + \
            l0_norm(self.linear.bias.unsqueeze(-1), self.log_alpha, *self.hyperparams).sum()

    @property
    def n_gates(self):
        return self.log_alpha.numel()

    def count_active(self):
        _, gamma, zeta = self.hyperparams
        return ((self.log_alpha.sigmoid() * (zeta - gamma) + gamma).clamp_(min=0) != 0).long().sum()

    def _transform_weight(self):
        return l0_weight_transform(self.linear.weight, self.log_alpha, *self.hyperparams, self.training)

    def _transform_bias(self):
        return l0_weight_transform(self.linear.bias.unsqueeze(-1), self.log_alpha, *self.hyperparams, self.training).squeeze(-1)

    def forward(self, x):
        weight = self._transform_weight()
        bias = None
        if self.linear.bias is not None:
            bias = self._transform_bias()
        return F.linear(x, weight, bias=bias)


class L0Conv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hyperparams = kwargs.get("beta", 2/3), kwargs.get("gamma", -0.1), kwargs.get("zeta", 1.1)
        self.conv = nn.Conv2d(*args, **kwargs)
        self.log_alpha = nn.Parameter(torch.empty(self.conv.weight.size(0)).normal_(0, 0.01))
        self.kwargs = kwargs
        try:
            del self.kwargs["kernel_size"]
        except:
            pass

    @property
    def l0_norm(self):
        return l0_norm(self.conv.weight, self.log_alpha, *self.hyperparams).sum() + \
            l0_norm(self.conv.bias.unsqueeze(-1), self.log_alpha, *self.hyperparams).sum()

    @property
    def n_gates(self):
        return self.log_alpha.numel()

    def count_active(self):
        _, gamma, zeta = self.hyperparams
        return ((self.log_alpha.sigmoid() * (zeta - gamma) + gamma).clamp_(min=0) != 0).long().sum()

    def _transform_weight(self):
        return l0_weight_transform(self.conv.weight, self.log_alpha, *self.hyperparams, self.training)

    def _transform_bias(self):
        return l0_weight_transform(self.conv.bias.unsqueeze(-1), self.log_alpha, *self.hyperparams, self.training).squeeze(-1)

    def forward(self, x):
        weight = self._transform_weight()
        bias = None
        if self.conv.bias is not None:
            self.kwargs["bias"] = self._transform_bias()
        return F.conv2d(x, weight, **self.kwargs)


def collect_nonzero_stats(x, stats=None):
    if stats is None:
        stats = torch.zeros(x.size(1)).byte().to(x.device)
    x = x.view(x.size(0), x.size(1), np.prod(x.size()[2:]))
    x = x.sum(0).sum(-1)
    stats |= (x != 0)
    return stats


def prune_conv(conv, stats, order="after"):
    if order == "after":
        conv.weight.data = conv.weight.data[stats].detach().contiguous()
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[stats].detach().contiguous()
    else:
        conv.weight.data = conv.weight.data[:, stats].detach().contiguous()


def prune_bn(bn, stats):
    if bn.track_running_stats:
        bn.running_mean = bn.running_mean[stats]
        bn.running_var = bn.running_var[stats]
    if bn.affine:
        bn.weight.data = bn.weight.data[stats]
        bn.bias.data = bn.bias.data[stats]


if __name__ == "__main__":
    weights = torch.ones(10, 320).cuda()
    weights.requires_grad = True
    log_alpha = torch.empty(10).normal_(0, 0.01).cuda()
    log_alpha.requires_grad = True
    norm = l0_norm(weights, log_alpha, 1/2, -0.1, 1.1)
    norm.sum().backward()
    print("log alpha grad:", log_alpha.grad)

    weights = torch.ones(10, 320).cuda()
    weights.requires_grad = True
    print("old weights:", weights[:, 0])
    gated_weights = l0_weight_transform(weights, log_alpha, 2/3, -0.1, 1.1)
    print("gated weights:", gated_weights[:, 0])
    y = F.linear(gated_weights, torch.ones(1, 320).cuda())
    y.sum().backward()
    print("weights grad:", weights.grad)