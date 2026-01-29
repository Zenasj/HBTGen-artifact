# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn
from torch._jit_internal import BroadcastingList4

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pad_op = nn.ZeroPad2d(padding=(10, 20, 0, 0))

    def forward(self, x):
        return self.pad_op(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 100, 100  # Example dimensions
    return torch.rand(B, C, H, W, dtype=torch.float32)

