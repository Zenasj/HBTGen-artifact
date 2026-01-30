py
import torch

class CustomFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a):
        return torch.exp_(a)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


a = torch.rand(2).requires_grad_()
fn = torch.compile(CustomFn.apply)
out = fn(a)
out.backward(torch.ones_like(out))