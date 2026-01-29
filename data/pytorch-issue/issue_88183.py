# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (1, 1, 1, 2)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.clone(memory_format=torch.contiguous_format)  # Ensure contiguous memory format
        x = torch.nn.functional.interpolate(x, size=[2, 2], mode="bilinear", align_corners=False)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.ones([1, 1, 1, 2], device="cpu")

