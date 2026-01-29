# torch.rand(1, 1, 1, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare data pointers and metadata between original and detached tensor
        original = x
        detached = x.detach()
        same_data = (
            original.data_ptr() == detached.data_ptr() and
            original.stride() == detached.stride() and
            original.shape == detached.shape
        )
        return torch.tensor([same_data], dtype=torch.long)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 3, dtype=torch.float32)

