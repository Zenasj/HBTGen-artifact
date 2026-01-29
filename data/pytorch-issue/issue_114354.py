# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, image):
        num_channels = image.shape[-3]
        kernel = torch.rand(num_channels, 1, 3, 3, device=image.device, dtype=image.dtype)
        return torch.nn.functional.conv2d(image, kernel, groups=num_channels)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 16, 16, dtype=torch.float32)

