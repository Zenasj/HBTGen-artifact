# torch.rand(B, C, H, W, dtype=...)  # Assuming B, C, H, W are the dimensions of the input tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        y = x.abs().max()
        z = x / 10.0
        z_t = z.t().contiguous().t()
        return y, z, z_t

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 3072, 4096, 1, 1  # Example dimensions, adjust as needed
    return torch.randn(B, C, device="cuda") / 10.0

