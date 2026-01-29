# torch.rand(2, 3, 40, 50, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Slice along width dimension (dim=3) as per test case
        # Example parameters: start=10, end=30, step=1 (default step)
        return x[:, :, :, 10:30]

def my_model_function():
    return MyModel()

def GetInput():
    # Returns 4D tensor matching Vulkan slice test case dimensions
    return torch.rand(2, 3, 40, 50, dtype=torch.float32)

