# torch.rand(1, 4, 20, 20, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 4, 20, 20, dtype=torch.float32, device='cuda')

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

