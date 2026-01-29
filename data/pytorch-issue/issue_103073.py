# torch.rand(10, 100000, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x is a tensor of shape (10, N)
        r = x[0].clone()  # Start with first tensor
        for idx, t in enumerate(x):
            r += (t * t + idx) / 2
        return r

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 100000, dtype=torch.float32)

