# torch.rand(B, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 2)  # Matches the original Linear model in the issue

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()  # Returns the base model instance

def GetInput():
    B = 1  # Inferred batch size from Linear(1,2) usage in the issue's example
    return torch.rand(B, 1, dtype=torch.float32)  # Matches input shape for Linear(1,2)

