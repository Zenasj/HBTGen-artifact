# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # This indexing triggers the __getitem__ issue when wrapped
        indices = torch.tensor([0, 1], device=x.device)
        return x[indices]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32)

