# torch.rand(1, 1, 6, 6, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = F.fractional_max_pool2d(x, kernel_size=3, output_size=(1, 1))
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 1, 6, 6, dtype=torch.float32)

