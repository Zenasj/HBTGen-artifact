# (torch.rand(1), torch.rand(1))  # Input shape: tuple of two tensors of shape (1,)
from collections import namedtuple
import torch
from torch import nn

Outputs = namedtuple('Outputs', ['val'])

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.model2 = Model2()
    
    def forward(self, inputs):
        x, y = inputs
        out1 = self.model1(x, y)
        out2 = self.model2(x)
        expected1 = 2 * x + y
        actual1 = out1.val
        expected2 = 2 * x
        actual2 = out2.val
        discrepancy = (actual1 != expected1).any() | (actual2 != expected2).any()
        return discrepancy.float()  # Returns 1.0 if discrepancy exists, else 0.0

class Model1(nn.Module):
    def forward(self, x, y):
        out = Outputs(val = x + y)
        out.val.add_(x)
        return out

class Model2(nn.Module):
    def forward(self, x):
        out = Outputs(val = x)
        out.val.add_(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.tensor([1.0])
    y = torch.tensor([3.0])
    return x, y

