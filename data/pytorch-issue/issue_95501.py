# Input is a tuple of four tensors of shape (3, 3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(3, 3))

    def forward(self, inputs):
        a, b, c, d = inputs
        a.t_()
        b.t_()
        c.t_()
        d.t_()
        return a + b + c + d + self.mean

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.rand(3, 3, requires_grad=True),
        torch.rand(3, 3, requires_grad=True),
        torch.rand(3, 3, requires_grad=True),
        torch.rand(3, 3, requires_grad=True)
    )

