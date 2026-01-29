# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Perform in-place zeroing on the second element of the input tensor
        x[1].zero_()
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random tensor of shape (3,) as input
    return torch.rand(3)

