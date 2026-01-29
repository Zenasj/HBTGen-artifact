# torch.rand(3, 4, dtype=torch.float32)
import torch
from torch import nn, Tensor
from torch.autograd import Function

class _SafeTanh(Function):
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
    def backward(ctx, grad_output):
        grad = grad_output
        output, = ctx.saved_tensors
        return grad * (1 - output.pow(2)), None  # No gradient for eps

class HalfRopeInPlace(Function):
    @staticmethod
    def forward(t: Tensor, freqs_cis: Tensor, conj: bool):
        # Dummy implementation (original _halfrope_inplace not provided)
        return t

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, freqs_cis, conj = inputs
        ctx.save_for_backward(freqs_cis)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        freqs_cis, = ctx.saved_tensors
        # Dummy backward mirroring the issue's implementation
        return grad_output, None, None

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotations = nn.Parameter(torch.randn(()))  # Example parameter for freqs_cis

    def forward(self, x):
        # Apply SafeTanh with fixed eps=1.0
        x = _SafeTanh.apply(x, 1.0)
        # Apply HalfRopeInPlace with stored rotations and conj=False
        return HalfRopeInPlace.apply(x, self.rotations, False)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4, dtype=torch.float32)

