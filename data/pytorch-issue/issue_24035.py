# torch.rand(n, m, dtype=torch.bool)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.triu_()

def my_model_function():
    return MyModel()

def GetInput():
    # Example dimensions (n=3, m=4). Adjust as needed.
    return torch.ones(3, 4, dtype=torch.bool)

