# torch.rand(1, 1, 4, 4, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Use AvgPool2d to avoid ONNX export issue

    def forward(self, x):
        x = self.avg_pool(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 4, 4, dtype=torch.float32)

