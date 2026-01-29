# torch.rand(1, 8, 8, 8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = torch.sigmoid(x)
        return torch.mean(x, dim=[-1, -2], keepdim=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 8, 8, 8, dtype=torch.float32)

