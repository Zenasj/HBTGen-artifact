# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return x.sum(dim=(2, 3), keepdim=True)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Inferred input shape: (B, C, H, W)
    B, C, H, W = 16, 3, 256, 256
    return torch.rand(B, C, H, W, dtype=torch.float32)

