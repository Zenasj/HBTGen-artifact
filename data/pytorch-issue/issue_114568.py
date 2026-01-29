import torch
import torch.nn as nn

# torch.rand(3, 3, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.pinverse(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)

