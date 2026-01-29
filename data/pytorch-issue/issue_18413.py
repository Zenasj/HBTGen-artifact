# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

def torch_pad_reflect(image: torch.Tensor, paddings: Sequence[int]) -> torch.Tensor:
    paddings = np.array(paddings, dtype=int)
    assert np.all(np.array(image.shape[-2:]) > 1), "Image shape should be more than 1 pixel"
    assert np.all(paddings >= 0), "Negative paddings not supported"

    while np.any(paddings):
        image_dims = image.shape[-2:]  # (H, W)
        # Compute maximum possible padding for each direction
        image_limits = np.array([
            image_dims[1]-1, image_dims[1]-1,  # W-1 for left/right
            image_dims[0]-1, image_dims[0]-1   # H-1 for top/bottom
        ])
        possible_paddings = np.minimum(paddings, image_limits)

        image = torch.nn.functional.pad(image, tuple(possible_paddings), mode='reflect')
        paddings = paddings - possible_paddings

    return image

class CustomReflectPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding  # (left, right, top, bottom)

    def forward(self, x):
        return torch_pad_reflect(x, self.padding)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Problematic padding (46+46 exceeds input width of 36)
        self.padding_layer = CustomReflectPad2d(padding=(46, 46, 10, 10))
        # Example convolution layer after padding
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=0)

    def forward(self, x):
        x = self.padding_layer(x)
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 128, 36, dtype=torch.float32)

