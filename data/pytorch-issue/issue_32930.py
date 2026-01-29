# torch.rand(10, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class ReturnTuple(nn.Module):
    def forward(self, x):
        return x * 3, x * 4, x * 5

class ABC(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = ReturnTuple()

    def forward(self, x):
        a, b, c = self.l(x)
        return a.relu(), b.sigmoid(), c.softmax(1)

class CBA(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = ReturnTuple()

    def forward(self, x):
        a, b, c = self.l(x)
        return c.relu(), b.sigmoid(), a.softmax(1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.abc = ABC()
        self.cba = CBA()

    def forward(self, x):
        abc_out = self.abc(x)
        cba_out = self.cba(x)
        # Compare first and third outputs for differences
        diff_first = not torch.allclose(abc_out[0], cba_out[0], atol=1e-5)
        diff_third = not torch.allclose(abc_out[2], cba_out[2], atol=1e-5)
        return torch.tensor([diff_first and diff_third], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, dtype=torch.float32)

