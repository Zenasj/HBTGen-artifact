# torch.rand(2, 4, 4, 4, dtype=torch.float32)  # Inferred input shape from C++ example's frame structure
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Returns the first element of the first "frame" tensor in the input structure
        return x[0, 0]

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the C++ example's input structure (2 frames, 4 tensors per frame, each 4x4)
    return torch.rand(2, 4, 4, 4, dtype=torch.float32)

