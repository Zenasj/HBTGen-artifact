# torch.rand(100, 200, dtype=torch.float32)  # Inferred input shape from operations
import torch
import torch.nn as nn

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        getitem = args[0]
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        mul = torch.mul(getitem, getattr_1)
        flatten = getattr_1.flatten()
        sum_1 = flatten.sum(0)
        to = flatten.to(dtype=torch.bool)
        return (mul, sum_1, to)

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        getitem = args[0]
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        mul = torch.mul(getitem, getattr_1)
        flatten = getattr_1.flatten()
        sum_1 = flatten.sum(0)
        to = flatten.to(dtype=torch.bool)
        return (to, sum_1, mul)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = Model0()
        self.model1 = Model1()

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        a, b, c = out0
        c1, b1, a1 = out1

        # Compare outputs using rtol=1.0 and exact equality for booleans
        rel_diff_a = torch.abs(a - a1) / torch.maximum(torch.abs(a), torch.abs(a1))
        rel_diff_b = torch.abs(b - b1) / torch.maximum(torch.abs(b), torch.abs(b1))
        
        close_a = torch.all(rel_diff_a <= 1.0)
        close_b = torch.all(rel_diff_b <= 1.0)
        close_c = torch.all(c == c1)

        return close_a & close_b & close_c

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 200, dtype=torch.float32)

