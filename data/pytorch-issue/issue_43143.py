# torch.rand(2, 2, dtype=torch.complex64)  # Using complex64 as a workaround for the complex32 issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Since complex32 is not supported, we use complex64 as a workaround
        self.linear = nn.Linear(2, 2, dtype=torch.complex64)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 2, dtype=torch.complex64)

