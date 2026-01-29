# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(64, 3, (3, 3))

    def forward(self, inp):
        if inp.is_contiguous(memory_format=torch.contiguous_format):
            inp = inp.to(memory_format=torch.channels_last)
        return self.conv2d(inp)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 64, 28, 28  # Example batch size, channels, height, width
    return torch.rand(B, C, H, W, dtype=torch.float32)

