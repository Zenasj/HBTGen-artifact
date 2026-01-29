# torch.rand(B, 1, H, W, dtype=torch.float32)  # Assuming a generic input shape (B, C, H, W) with C=1 for simplicity

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        x = torch.neg(x)
        # foo foo foo i have a comment at the wrong indent
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming a batch size of 4, 1 channel, and image dimensions of 64x64
    B, C, H, W = 4, 1, 64, 64
    return torch.rand(B, C, H, W, dtype=torch.float32)

