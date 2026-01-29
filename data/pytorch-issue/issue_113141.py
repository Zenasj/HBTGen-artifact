# torch.randint(0, 256, (40, 30), dtype=torch.uint8)
import torch
from torch import nn

class Model0(torch.nn.Module):
    def __init__(self, v4_0, v1_0):
        super().__init__()
        self.v4_0 = v4_0
        self.v1_0 = v1_0

    def forward(self, *args):
        _args = args
        v4_0 = self.v4_0
        v1_0 = self.v1_0
        getitem = _args[0]
        reshape = v1_0.reshape(30)
        neg = torch.neg(v4_0)
        add = torch.add(reshape, getitem)
        mul = torch.mul(add, neg)
        sum_1 = mul.sum(0)
        return (sum_1,)

class Model1(torch.nn.Module):
    def __init__(self, v4_0, v1_0):
        super().__init__()
        self.v4_0 = v4_0
        self.v1_0 = v1_0

    def forward(self, *args):
        _args = args
        v4_0 = self.v4_0
        v1_0 = self.v1_0
        getitem = _args[0]
        reshape = v1_0.reshape(30)
        neg = torch.neg(v4_0)
        add = torch.add(reshape, getitem)
        mul = torch.mul(neg, add)
        sum_1 = mul.sum(0)
        return (sum_1,)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v4_0 = nn.Parameter(torch.empty(40, 30, dtype=torch.uint8), requires_grad=False)
        self.v1_0 = nn.Parameter(torch.empty(30, dtype=torch.uint8), requires_grad=False)
        self.v4_0.data.random_(0, 256)  # Initialize parameters with random values
        self.v1_0.data.random_(0, 256)
        self.model0 = Model0(self.v4_0, self.v1_0)
        self.model1 = Model1(self.v4_0, self.v1_0)
    
    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        are_different = not torch.allclose(out0[0], out1[0], rtol=1, atol=0)
        return torch.tensor(are_different, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 256, (40, 30), dtype=torch.uint8)

