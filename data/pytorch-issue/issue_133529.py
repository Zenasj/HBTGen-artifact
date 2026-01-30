from torch.autograd import Function
import torch

class _SafeTanh(Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input, eps):
        output = input.tanh()
        lim = 1.0 - eps
        output = output.clamp(-lim, lim)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (output,) = ctx.saved_tensors
        return (grad * (1 - output.pow(2)), None)

func = _SafeTanh.apply

x = torch.randn((1, 1), requires_grad=True)
eager = lambda x, y: torch.vmap(lambda x: func(x, y))
eager(x, 1.0)
c = torch.compile(eager, fullgraph=True)
y = c(x, 1.0)
eager = torch.vmap(lambda x: func(x, 1.0))
eager(x)
c = torch.compile(eager, fullgraph=True)
y = c(x)

class HalfRopeInPlace(torch.autograd.Function):
    @staticmethod
    def forward(t: Tensor, freqs_cis: Tensor, conj: bool):
        _halfrope_inplace(t, freqs_cis, conj)
        return t

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, freqs_cis, conj = inputs
        ctx.save_for_backward(freqs_cis)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (freqs_cis,) = ctx.saved_tensors
        _halfrope_inplace(grad_output, freqs_cis, not ctx.conj)
        return grad_output, None, None

HalfRopeInPlace.apply(qk.unflatten(-1, ((-1, 2))), rotations, False)

class HalfRopeInPlace(torch.autograd.Function):
    @staticmethod
    def forward(*args):
        *_, t, freqs_cis, conj = args
        _halfrope_inplace(t, freqs_cis, conj)
        return t

import torch
from torch import Tensor

def _halfrope_inplace(*args):
    ...

class HalfRopeInPlace(torch.autograd.Function):
    @staticmethod
    def forward(t: Tensor, freqs_cis: Tensor, conj: bool):
        _halfrope_inplace(t, freqs_cis, conj)
        return t

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, freqs_cis, conj = inputs
        ctx.save_for_backward(freqs_cis)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (freqs_cis,) = ctx.saved_tensors
        _halfrope_inplace(grad_output, freqs_cis, not ctx.conj)
        return grad_output, None, None

t = torch.randn(3, 4)
rotations = torch.randn(())

@torch.compile(fullgraph=True)
def func(t, rotations):
    return HalfRopeInPlace.apply(t, rotations, False)

func(t, rotations)