# torch.rand(B, C, 1, 1, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[4.0, 5.0, 6.0]])
        c = torch.cat((a, b), 0)
        return b + c

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 1, 1, dtype=torch.float)

