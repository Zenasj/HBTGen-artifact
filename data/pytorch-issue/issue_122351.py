# torch.rand(5, 2, 2, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(5, 2, 2, dtype=torch.float32)

