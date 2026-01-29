# torch.rand(B, 3, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 10)  # Matches input features (C=3)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 4D input to 2D (B, 3)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 3, 1, 1)  # Matches original example input (5x3) as 4D tensor

