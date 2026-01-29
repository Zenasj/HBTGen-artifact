# torch.rand(8, 48, 24, 16, dtype=torch.float32)  # Inferred input shape based on the provided tensor sizes

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(48, 192, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(8, 48, 24, 16, dtype=torch.float32)

