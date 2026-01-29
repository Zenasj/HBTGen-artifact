import torch
from torch import nn

# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape is (1, 1, 1, 1)
class MyModel(nn.Module):
    def forward(self, x):
        # Use as_strided as suggested in the workaround comment to avoid set_() issues
        return x.as_strided(x.size(), x.stride(), storage_offset=0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

