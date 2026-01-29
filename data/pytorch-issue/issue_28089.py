# torch.rand(1, 3, 4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compute variance over dimensions 2 and 3 (height and width)
        # This requires PyTorch version >=1.2.0 where dim can be a tuple
        return torch.var(x, dim=(2, 3), keepdim=True)

def my_model_function():
    # Returns a model that computes channel-wise variance over H/W dimensions
    return MyModel()

def GetInput():
    # Returns a random input tensor matching (Batch, Channels, Height, Width)
    return torch.rand(1, 3, 4, 4, dtype=torch.float32)

