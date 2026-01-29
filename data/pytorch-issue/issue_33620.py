# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute element size based on dtype to avoid using unsupported element_size()
        elem_size = 0
        if x.dtype == torch.float32:
            elem_size = 4
        elif x.dtype == torch.float64:
            elem_size = 8
        elif x.dtype == torch.int32:
            elem_size = 4
        else:
            elem_size = 4  # Default to 4 bytes if unknown (assumption)
        total_bytes = x.numel() * elem_size
        return torch.tensor([total_bytes], dtype=torch.int64)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

