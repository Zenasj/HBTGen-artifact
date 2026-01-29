# torch.rand(1, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        t1 = torch.add(x, x)
        t2 = t1.unfold(1, 3, 2)
        t3 = t2.abs_()
        return t3

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([1, 5], dtype=torch.float32)

