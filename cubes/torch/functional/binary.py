import numpy as np
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F


def sech2(x):
    return 1 - x.tanh()**2


class BinaryActivation(nn.Module):

    def __init__(self, t=0., soft=True, stochastic=False):
        super().__init__()
        self.soft = soft
        self.scale = t
        self.stochastic = stochastic

    def forward(self, x):
        if self.soft:
            return binary_tanh(x, self.scale, stochastic=self.stochastic)
        else:
            return binary_tanh(x, stochastic=self.stochastic)


def hard_sigmoid(x):
    return inclusive_clamp(((x + 1) / 2), 0, 1)


def binary_tanh(x, scale=1, stochastic=False):
    if stochastic:
        return 2 * soft_bernoulli(x.sigmoid(), scale) - 1
    else:
        return 2 * soft_round(hard_sigmoid(x), scale) - 1


class SoftRoundFunction(ag.Function):

    @staticmethod
    def forward(ctx, input, scale=1):
        if scale < 1:
            return scale * input.round() + (1 - scale) * input
        else:
            return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class SoftBernoulliFunction(ag.Function):

    @staticmethod
    def forward(ctx, alpha, scale=1):
        if scale < 1:
            return scale * alpha.bernoulli() + (1 - scale) * alpha
        else:
            return torch.bernoulli(alpha)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class InclusiveClamp(ag.Function):

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x)
        ctx.limit = (a, b)
        return x.clamp(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        a, b = ctx.limit
        grad_output[(x < a) | (x > b)] = 0
        return grad_output, None, None


def logb2(x):
    return torch.log(x) / np.log(2)


def ap2(x):
    x = x.sign() * torch.pow(2, torch.round(logb2(x.abs())))
    return x


class ApproxPow2Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ap2(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_variables
        return grad_output


class BinaryResidualAddFunction(ag.Function):

    @staticmethod
    def forward(ctx, a, b):
        random_mask = (a != b)
        ctx.save_for_backward(random_mask)
        mask = (random_mask.float() * 0.5).bernoulli()
        res = a / 2 + b / 2
        res[random_mask] = res[random_mask].fill_(0.5).bernoulli_().sub_(0.5).mul_(2)
        return res

    @staticmethod
    def backward(ctx, grad):
        random_mask, = ctx.saved_variables
        grad[random_mask] = 0
        return grad / 2, grad / 2


class BinaryReLUFunction(ag.Function):

    @staticmethod
    def forward(ctx, x):
        random_mask = (x < 0)
        ctx.save_for_backward(random_mask)
        x[random_mask] = x[random_mask].fill_(0.5).bernoulli_().sub_(0.5).mul_(2)
        return x

    @staticmethod
    def backward(ctx, grad):
        random_mask, = ctx.saved_variables
        grad[random_mask] = 0
        return grad


class PassThroughDivideFunction(ag.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_variables
        return grad_output, -grad_output * x / y**2


class QuantizedBatchNorm(nn.Module):

    def __init__(self, num_features, momentum=0.125, eps=1e-5, affine=True, k=1, min=-1, max=1):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.k = k
        self.min = min
        self.max = max
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.reset_parameters()
        self._init_quantize_fn()

    def _init_quantize_fn(self):
        self.quantize_fn = approx_pow2

    def reset_parameters(self):
        self.mean.zero_()
        self.var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.weight.data -= 0.5
            self.bias.data.zero_()

    def _convert_param(self, x, param):
        param = param.repeat(x.size(0), 1)
        for _ in x.size()[2:]:
            param = param.unsqueeze(-1)
        param = param.expand(-1, -1, *x.size()[2:])
        return param

    def _reorg(self, input):
        axes = [1, 0]
        axes.extend(range(2, input.dim()))
        return input.permute(*axes).contiguous().view(input.size(1), -1)

    def forward(self, input):
        if self.training:
            new_mean = self._reorg(input).mean(1).data
            self.mean = self.quantize_fn((1 - self.momentum) * self.mean + self.momentum * new_mean)
        mean = self._convert_param(input, self.mean)
        ctr_in = self.quantize_fn(input - mean)

        if self.training:
            new_var = self._reorg(ctr_in * ctr_in).mean(1).data
            self.var = self.quantize_fn((1 - self.momentum) * self.var + self.momentum * new_var)
        var = self._convert_param(input, self.var)
        x = self.quantize_fn(ctr_in / self.quantize_fn(torch.sqrt(var + self.eps)))

        if self.affine:
            w1 = self._convert_param(x, self.weight)
            b1 = self._convert_param(x, self.bias)
            y = self.quantize_fn(self.quantize_fn(w1) * x + self.quantize_fn(b1))
        else:
            y = x
        return y


approx_pow2 = ApproxPow2Function.apply
passthrough_div = PassThroughDivideFunction.apply
soft_round = SoftRoundFunction.apply
inclusive_clamp = InclusiveClamp.apply
soft_bernoulli = SoftBernoulliFunction.apply
binary_residual_add = BinaryResidualAddFunction.apply
binary_relu = BinaryReLUFunction.apply


class BinaryLinear(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.linear = nn.Linear(*args)
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return F.linear(x, binary_tanh(self.linear.weight), bias=binary_tanh(self.linear.bias) if self.linear.bias is not None else None)


class BinaryConv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.conv.weight.data.normal_(0, 0.01)
        self.kwargs = kwargs
        try:
            del self.kwargs["kernel_size"]
        except:
            pass

    def forward(self, x):
        self.kwargs["bias"] = binary_tanh(self.conv.bias) if self.conv.bias is not None else None
        return F.conv2d(x, binary_tanh(self.conv.weight), **self.kwargs)
