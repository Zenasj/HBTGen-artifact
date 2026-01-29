# torch.rand(1, 1, 5, 5, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a MaxPool2d layer with valid padding
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=2)

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 1, 5, 5)

