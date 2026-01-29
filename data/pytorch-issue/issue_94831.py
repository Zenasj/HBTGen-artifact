# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        x.add_(1)  # In-place addition
        target_size = torch.randn(3, 4)  # Create temporary tensor for resize
        x.resize_as_(target_size)  # In-place resize to (3,4)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)

