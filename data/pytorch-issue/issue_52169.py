# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, scale, dtype_min, dtype_max):
        super(MyModel, self).__init__()
        self.scale = scale
        self.dtype_min = dtype_min
        self.dtype_max = dtype_max

    def forward(self, x):
        f = x / self.scale
        f = torch.round(f)
        f = f.clamp(self.dtype_min, self.dtype_max)
        f = f * self.scale
        return f

def my_model_function():
    # Example parameters based on quantization context (common for 8-bit)
    scale = torch.tensor(1.0)  # Example scale value
    dtype_min = 0.0
    dtype_max = 255.0
    return MyModel(scale, dtype_min, dtype_max)

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

