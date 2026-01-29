import torch
from torch import nn

class CustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i + i
        tmp = (result, i)
        ctx.cached_data = tmp
        return result

    @staticmethod
    def backward(ctx, grad_output):
        for t in ctx.cached_data:
            if hasattr(t, 'saved_data'):
                result = grad_output * t
            else:
                result = grad_output * 2
        ctx.cached_data = None
        return result

# torch.rand(2, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, x):
        return CustomOp.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32, requires_grad=True)

