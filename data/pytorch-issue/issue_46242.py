# torch.rand(2, 8, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 16)
        self.token = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        # Convert 4D input (B, C, 1, 1) to 2D (B, C)
        x = x.view(x.size(0), -1)
        x[[True, False]] = self.token  # Assign token to selected batch elements
        x = self.layer(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 8, 1, 1)  # Matches input shape expectation

