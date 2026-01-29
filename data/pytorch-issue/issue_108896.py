# torch.rand(4, dtype=torch.float32, device='cuda')  # Inferred input shape from the issue's examples
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x + 1  # Matches the operation inside the CUDA graph in the issue's examples

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32, device='cuda')

