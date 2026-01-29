# torch.rand(1, 1, 1, 2**20, dtype=torch.float32)  # Inferred from example input in the issue
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.sum()

def my_model_function():
    return MyModel()

def GetInput():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.rand(1, 1, 1, 2**20, dtype=torch.float32, device=device)

