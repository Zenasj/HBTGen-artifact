# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Output features set to 0 to reproduce the shape causing division by zero
        self.fc = nn.Linear(3, 0)  # Creates output shape (N, 0)

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input that produces output shape (5, 0) when passed through MyModel
    return torch.rand(5, 3, dtype=torch.float32)

