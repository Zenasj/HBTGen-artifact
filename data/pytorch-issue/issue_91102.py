import torch
from torch import nn

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

# torch.rand(10, dtype=torch.float32, requires_grad=True)
class MyModel(nn.Module):
    def forward(self, x):
        return Exp.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, requires_grad=True)

