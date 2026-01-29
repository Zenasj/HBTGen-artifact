# torch.rand(1, 2, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.minimum(x, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 2, 224, 224)

