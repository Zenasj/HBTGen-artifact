# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, upscale_factor=1):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

def my_model_function():
    return MyModel(upscale_factor=1)

def GetInput():
    return torch.rand(1, 9, 2, 2)

