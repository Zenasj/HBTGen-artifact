# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.sum()  # Matches the operation causing the original issue (sum of tensor with inf)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with a single element to match the issue's input pattern
    # Uses random value but ensures dtype matches the original Tensor() usage (float32)
    return torch.rand(1, dtype=torch.float32)

