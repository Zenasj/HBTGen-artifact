# torch.rand(20, 16, 50, 32, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the FractionalMaxPool2d layer with correct parameters
        self.fractional_max_pool = torch.nn.FractionalMaxPool2d(kernel_size=(2, 1), output_ratio=(0.5, 0.5))

    def forward(self, x):
        return self.fractional_max_pool(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(20, 16, 50, 32, dtype=torch.float32)

