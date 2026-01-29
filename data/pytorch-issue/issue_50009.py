# torch.rand(1, 4, 1, 80, dtype=torch.float32)  # Inferred input shape from error context (4x4 weight and common TTS dimensions)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_splits=4):
        super().__init__()
        # Create a parameter for the problematic weight with requires_grad=False
        self.weight = nn.Parameter(torch.randn(num_splits, num_splits), requires_grad=False)
        self.num_splits = num_splits

    def forward(self, x):
        # Replicate the problematic weight reshaping and convolution
        weight = self.weight.view(self.num_splits, self.num_splits, 1, 1)
        return F.conv2d(x, weight)

def my_model_function():
    # Initialize with num_splits=4 as seen in the error log
    return MyModel(num_splits=4)

def GetInput():
    # Generate input matching the 4D shape expected by the model
    return torch.rand(1, 4, 1, 80, dtype=torch.float32)

