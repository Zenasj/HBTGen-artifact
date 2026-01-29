# torch.rand(16, 3, 100, 250, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.block_1 = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x):
        x = self.block_1(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(16, 3, 100, 250, dtype=torch.float32)

