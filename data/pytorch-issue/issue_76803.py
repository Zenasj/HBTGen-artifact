# torch.rand(2, 2, dtype=torch.cfloat), torch.tensor(1, dtype=torch.double), torch.rand(2, 2, dtype=torch.float), torch.tensor(1, dtype=torch.cdouble)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a1, b1, a2, b2 = inputs
        res1 = a1 + b1
        res2 = a2 + b2
        expected1 = torch.complex128
        expected2 = torch.complex64
        correct1 = torch.tensor(res1.dtype == expected1, dtype=torch.bool)
        correct2 = torch.tensor(res2.dtype == expected2, dtype=torch.bool)
        return correct1 & correct2

def my_model_function():
    return MyModel()

def GetInput():
    a1 = torch.randn(2, 2, dtype=torch.cfloat)
    b1 = torch.tensor(1, dtype=torch.double)
    a2 = torch.randn(2, 2, dtype=torch.float)
    b2 = torch.tensor(1, dtype=torch.cdouble)
    return (a1, b1, a2, b2)

