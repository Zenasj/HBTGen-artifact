# torch.rand(B, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(5, 5),
            nn.Linear(5, 5),
        )
        self.eval()  # Matches original issue's model setup

    def forward(self, x):
        return self.seq(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Default batch size (can be adjusted)
    return torch.rand(B, 5, dtype=torch.float32)

