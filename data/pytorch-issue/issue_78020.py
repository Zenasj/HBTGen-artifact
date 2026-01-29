# torch.rand(1, 1, 1, 1, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Perform operations that trigger MPS bool dtype issues
        mul_result = x * 2  # MPS returns 255 for True, CPU returns 2
        add_result = x + 1  # MPS crashes here, CPU works
        return mul_result, add_result

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D bool tensor (B=1, C=1, H=1, W=1) with random True/False
    return torch.randint(0, 2, (1, 1, 1, 1), dtype=torch.bool)

