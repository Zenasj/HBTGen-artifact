# torch.rand(B, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(64, 64)  # Matches the Linear layer in the issue's repro code

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()  # Directly returns the model instance

def GetInput():
    # Generates input tensor matching the model's expected input shape (batch, 64)
    return torch.rand(2, 64, dtype=torch.float32)  # Batch size 2 as a safe default

