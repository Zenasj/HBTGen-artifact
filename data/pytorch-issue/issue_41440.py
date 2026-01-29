# torch.rand(1, 3, dtype=torch.float)
import torch
from torch import nn

class CustomFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op('Custom', input, outputs=2)
    @staticmethod
    def forward(ctx, input):
        return input, input

class MyModel(nn.Module):
    def forward(self, input):
        return CustomFunction.apply(input)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3)

