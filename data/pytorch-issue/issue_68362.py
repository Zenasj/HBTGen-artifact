# torch.rand(5, dtype=torch.float32, requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.pow(2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, dtype=torch.float32, requires_grad=True)

