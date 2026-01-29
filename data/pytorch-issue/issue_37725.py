# torch.rand(B, C, H, W, dtype=torch.float32, device='cuda').to(memory_format=torch.channels_last)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, groups=24, bias=False)
        self.conv1 = self.conv1.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = self.conv1(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 200, 24, 56, 56
    x = torch.rand(B, C, H, W, dtype=torch.float32, device='cuda').to(memory_format=torch.channels_last)
    return x

