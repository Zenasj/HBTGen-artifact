# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

