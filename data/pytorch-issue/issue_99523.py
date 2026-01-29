# torch.rand(3, dtype=torch.double, requires_grad=True)
import torch
from torch import nn

def my_square(x):
    return x**2

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        with torch.enable_grad():
            result = my_square(x)
        ctx.save_for_backward(x, result)
        return result.clone()

    @staticmethod
    def backward(ctx, grad_out):
        x, result = ctx.saved_tensors
        (grad_x,) = torch.autograd.grad(
            result,
            x,
            grad_outputs=grad_out,
            create_graph=True,
        )
        return grad_x

class MyModel(nn.Module):
    def forward(self, x):
        return Square.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.double, requires_grad=True)

