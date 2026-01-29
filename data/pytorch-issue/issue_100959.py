# torch.rand(1, 2, 14, 14, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.pool = nn.MaxPool2d(1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 2, 14, 14, dtype=torch.float32)

