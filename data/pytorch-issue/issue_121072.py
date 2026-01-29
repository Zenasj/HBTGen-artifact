# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape (B: batch size, C: channels, H: height, W: width)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)

    def forward(self, x):
        print("1 x dtype", x.dtype)
        x = self.conv(x)
        print("2 x dtype", x.dtype)
        x = F.interpolate(x, scale_factor=(2.5, 2.5), mode="bicubic", align_corners=False)
        print("3 x dtype", x.dtype)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 3, 32, 32, device="cuda", dtype=torch.float32)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# print(output)

