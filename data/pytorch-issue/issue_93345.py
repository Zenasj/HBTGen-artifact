# torch.rand(B, C, H, W, dtype=torch.float32)  # Assuming B=1, C=1, H=28, W=28 based on the example input
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        y = torch.quantize_per_tensor(x, 1.0, 0, torch.qint8)
        return y.int_repr()

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape and dtype
    return torch.rand(1, 1, 28, 28, dtype=torch.float32)

