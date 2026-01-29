# torch.rand(B, 3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3)  # B=1, 3 features as per the Linear(3,5) input requirements

