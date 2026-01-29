# torch.rand(3, 3, dtype=torch.float, requires_grad=True) for each of the two input tensors
import torch
from torch import nn
from torch.autograd import Function

class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        # Fixed implementation using autograd Function
        fixed_out = AddFunction.apply(a, b)
        # Simulated broken implementation (detaches to mimic requires_grad=False)
        broken_out = (a + b).detach().requires_grad_(False)
        # Return boolean indicating if requires_grad differs between outputs
        return torch.tensor(
            broken_out.requires_grad != fixed_out.requires_grad,
            dtype=torch.bool
        )

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(3, 3, dtype=torch.float, requires_grad=True)
    b = torch.rand(3, 3, dtype=torch.float, requires_grad=True)
    return (a, b)

