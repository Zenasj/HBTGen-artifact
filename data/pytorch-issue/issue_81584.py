# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 10, dtype=torch.float32)

