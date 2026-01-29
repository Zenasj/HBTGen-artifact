# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, dtype=torch.float32)

