# torch.rand(B, 1, L, dtype=torch.float32)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 4))  # Initialize kernel as a learnable parameter

    def forward(self, x):
        input_length = x.shape[-1]
        # Force parity to integer using explicit if/else to avoid symbolic types
        parity = 0 if input_length % 2 == 0 else 1
        padding = math.ceil((self.kernel.shape[-1] + parity) / 2) - 1
        return F.conv1d(x, self.kernel, padding=padding, stride=2)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with shape (1, 1, 249) that triggers dynamic shape behavior
    return torch.randn(1, 1, 249)

