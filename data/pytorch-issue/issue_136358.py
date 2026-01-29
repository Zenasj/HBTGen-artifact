# torch.rand(B, 1, 1, 1, dtype=torch.float32)  # Inferred input shape based on the repro example
import torch
from torch import nn

class IdentityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        class Node():
            pass

        a = Node()
        b = Node()
        a.b = b
        b.a = a  # Induce reference cycle

        s = torch.zeros(1,)
        s._attrs = {"key": "value"}
        a.s = s  # Tensor is part of the cycle
        ctx.save_for_backward(s)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MyModel(nn.Module):
    def forward(self, x):
        return IdentityFunction.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, requires_grad=True, dtype=torch.float32)

