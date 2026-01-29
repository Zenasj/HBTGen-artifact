# torch.rand(1, 3, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(3, 5, kernel_size=(49, 2), stride=(11, 11), padding=(1, 1))
        self.bn = nn.BatchNorm2d(5)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming a reasonable input size for demonstration purposes
    H, W = 50, 50
    return torch.rand(1, 3, H, W, dtype=torch.float32)

