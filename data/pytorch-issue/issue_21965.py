# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)  # Matches the Linear model in the issue's example

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generates input matching the Linear layer's expected shape (batch_size, 2)
    return torch.rand(1, 2, dtype=torch.float32)

