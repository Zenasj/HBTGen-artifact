# torch.rand(3,4), torch.rand(4,5), torch.rand(3,5)  # input shapes for x, y, inp
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y, inp = inputs
        return torch.add(torch.mm(x, y), inp)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    inp = torch.randn(3, 5)
    return (x, y, inp)

