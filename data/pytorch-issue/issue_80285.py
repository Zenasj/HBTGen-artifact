# torch.rand(4, 3, dtype=torch.uint8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.permute(1, 0)

def my_model_function():
    return MyModel()

def GetInput():
    # Replicates the exact input from the issue's example for reproducibility
    return torch.arange(4*3, dtype=torch.uint8).reshape(4, 3)

