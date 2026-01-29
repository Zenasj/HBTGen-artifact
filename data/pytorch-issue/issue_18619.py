# torch.rand(2, 4, 4, dtype=torch.float32)
import torch
from torch import nn

class Mult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.prod(dim=-1).prod(dim=-1)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return (grad_output * y)[:, None, None] / x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mult2 = Mult.apply  # Custom autograd function
        self.mult1 = lambda x: x.prod(dim=-1).prod(dim=-1)  # Equivalent to original mult1()

    def forward(self, x):
        # Return outputs of both functions for comparison
        return self.mult1(x), self.mult2(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from the issue's original test case
    return torch.rand(2, 4, 4, dtype=torch.float32).requires_grad_()

