# torch.rand(B, C, H, W, dtype=torch.float32)  # The input shape is inferred to be (B, C, H, W) with float32 dtype to avoid overflow issues

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No specific parameters or submodules needed for this simple model

    def forward(self, x):
        # Perform the rfft and return the result
        x_freq = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')
        return x_freq

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    size = 64
    B, C = 1, 1  # Assuming batch size and channel size
    x = torch.randn(B, C, size, size, size, dtype=torch.float32, device="cuda")
    return x

