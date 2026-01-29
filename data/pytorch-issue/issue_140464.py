# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_count = 0  # Trackable integer property
        
    def forward(self, x):
        return x * self.step_count

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)

