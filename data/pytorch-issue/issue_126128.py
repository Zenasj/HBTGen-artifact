# Inputs: (x1 (1, 10), y1 (10, 1), x2 (10, 100), y2 (100, 10))
import torch
from torch import nn

class MyModel1(nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y).sum()

class MyModel2(nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y).sum()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = MyModel1()
        self.model2 = MyModel2()

    def forward(self, inputs):
        x1, y1, x2, y2 = inputs
        out1 = self.model1(x1, y1)
        out2 = self.model2(x2, y2)
        return (out1, out2)

def my_model_function():
    return MyModel()

def GetInput():
    x1 = torch.randn(1, 10)
    y1 = torch.randn(10, 1)
    x2 = torch.randn(10, 100)
    y2 = torch.randn(100, 10)
    return (x1, y1, x2, y2)

