# torch.rand(2, 2, 2, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.axes = (1, 3)  # Axes from the issue's example

    def forward(self, x):
        # Compute nuclear norm using PyTorch's implementation
        return torch.norm(x, "nuc", dim=self.axes)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from the issue's example (after slicing)
    return torch.rand(2, 2, 2, 1, dtype=torch.float32)

