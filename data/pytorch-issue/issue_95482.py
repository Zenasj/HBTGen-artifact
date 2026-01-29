# torch.rand(2, 3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x[:1]  # Slice to (1, 3, 3)
        a = x[:, 0:1, 0:1]
        b = x[:, 1:2, 1:2]
        return a * b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3, 3, dtype=torch.float32)

