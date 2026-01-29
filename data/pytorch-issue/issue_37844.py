# torch.rand(1, 6, 20, 10, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.circular_conv = nn.Conv2d(6, 1, (3, 3), padding=(0, 1), padding_mode='circular')
        self.normal_conv = nn.Conv2d(6, 1, (3, 3), padding=(0, 1))

    def forward(self, x):
        circular_output = self.circular_conv(x)
        normal_output = self.normal_conv(x)
        return circular_output, normal_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 6, 20, 10, dtype=torch.float32)

