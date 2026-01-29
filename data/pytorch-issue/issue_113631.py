# torch.rand(21, 1, 1, 10, 34, dtype=torch.float32)
import torch
import torch.nn as nn

class Model0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        getitem = args[0]
        softmax = torch.softmax(getitem, dim=0)
        neg = torch.neg(softmax)
        add = torch.add(softmax, neg)
        return (add,)

class Model1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        getitem = args[0]
        softmax = torch.softmax(getitem, dim=0)
        transpose_1 = softmax.transpose(1, 0)
        neg = torch.neg(softmax)
        add = torch.add(softmax, neg)
        return (transpose_1, add)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = Model0()
        self.model1 = Model1()

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        add0 = out0[0]
        add1 = out1[1]
        is_close = torch.allclose(add0, add1, rtol=1e-07, atol=0.0)
        return torch.tensor(is_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(21, 1, 1, 10, 34, dtype=torch.float32)

