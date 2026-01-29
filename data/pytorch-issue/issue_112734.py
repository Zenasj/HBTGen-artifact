# torch.rand(B, C, D, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_max_pool3d_with_indices(
            input=x,
            output_size=self.output_size,
            return_indices=False
        )

def my_model_function():
    # Matches the original test case's output_size=2 (passed as first argument)
    return MyModel(output_size=2)

def GetInput():
    # Matches the input tensor shape from the original test case
    return torch.rand(9, 10, 9, 8, 6, dtype=torch.float32)

