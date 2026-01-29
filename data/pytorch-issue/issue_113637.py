# torch.rand(11, 24, 17, dtype=torch.uint8)
import torch
from torch import nn

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        max_1 = getitem.max(1)
        getattr_1 = max_1.values
        argmin = getattr_1.argmin(1)
        neg = torch.neg(getattr_1)
        to = neg.to(dtype=torch.int32)
        return (argmin, to)

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        max_1 = getitem.max(1)
        getattr_1 = max_1.values
        argmin = getattr_1.argmin(1)
        neg = torch.neg(getattr_1)
        to = neg.to(dtype=torch.int32)
        return (to, argmin)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = Model0()
        self.model1 = Model1()

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        argmin0, to0 = out0
        to1, argmin1 = out1
        argmin_eq = torch.all(argmin0 == argmin1)
        to_eq = torch.all(to0 == to1)
        return argmin_eq & to_eq  # Returns True if both outputs match exactly

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 255, (11, 24, 17), dtype=torch.uint8)

