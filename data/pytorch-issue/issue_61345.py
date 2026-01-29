# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Adjusted buffer shape to avoid slicing in forward pass
        self.register_buffer("rb", torch.randn(1, 3, 1, 1))  # Shape (1, C, 1, 1) for channel-wise addition

    def forward(self, x):
        return x + self.rb  # Direct addition without slicing

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

