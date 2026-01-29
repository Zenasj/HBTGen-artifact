# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (2, 3) for the given model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = nn.Linear(3, 3)
        self.linear_1 = nn.Linear(3, 3)

    def forward(self, x):
        return torch.sum(self.linear_1(self.linear_0(x)))

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand((2, 3), dtype=torch.float32)

