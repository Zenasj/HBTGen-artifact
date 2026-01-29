# torch.rand(B, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 5, dtype=torch.float32)

