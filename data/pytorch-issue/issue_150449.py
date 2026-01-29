# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(3, 1),
            nn.Flatten(0, 1)  # Matches the original model structure
        )
    
    def forward(self, x):
        return self.seq(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2000, 3, dtype=torch.float32)

