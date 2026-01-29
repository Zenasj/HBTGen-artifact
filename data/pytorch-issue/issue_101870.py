# torch.rand(1, 154828800, 1, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Reproduces the reduction operation that triggers the CUDA assertion error
        return x.mean(dim=-2)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor matching the shape and CUDA device used in the issue's example
    return torch.rand(1, 154828800, 1, 4, dtype=torch.float32).cuda()

