# torch.rand(128, 256, 56, 56, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.conv2(x)
        y = x2 * x
        return self.pool(y)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(128, 256, 56, 56, dtype=torch.float32)

