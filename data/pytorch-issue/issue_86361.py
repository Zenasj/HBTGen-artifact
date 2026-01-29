# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (1, 3, 600, 400) or (1, 1, 600, 400)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Example of using the interpolate function
        output_size = (224, 224)
        mode = 'nearest'  # Can be 'linear', 'nearest', 'nearest-exact'
        align_corners = None if "nearest" in mode else False
        return torch.nn.functional.interpolate(x, size=output_size, mode=mode, align_corners=align_corners)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 600, 400  # Example input shape
    dtype = torch.float32  # Example data type
    x = torch.rand(B, C, H, W, dtype=dtype)
    x = x.contiguous(memory_format=torch.channels_last)
    return x

