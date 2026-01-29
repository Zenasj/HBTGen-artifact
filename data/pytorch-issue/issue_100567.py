# torch.rand(100, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(100, 100)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.l(x))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, dtype=torch.float32)

