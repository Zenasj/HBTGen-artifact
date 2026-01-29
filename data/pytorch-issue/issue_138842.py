# torch.rand(B, 2, dtype=torch.float32)  # Input shape for MyModel
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 2, dtype=torch.float32)  # Example batch size 32

