# torch.rand(10, dtype=torch.float32)
import torch
from torch import nn, autograd

return_nan = True  # Control variable from the original code

class MyBadFn(autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return inp ** 2

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        res = 2 * grad_out * inp
        if return_nan:
            res = grad_out / 0 * 0  # Intentional NaN
        return res

class MyModel(nn.Module):
    def forward(self, x):
        return MyBadFn.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32, requires_grad=True)

