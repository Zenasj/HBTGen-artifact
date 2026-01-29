# torch.rand(1, 3, 224, 224, dtype=torch.int32)
import torch
from torch import nn

def has_integer_dtype(tensor: torch.Tensor, signed: bool | None = None) -> bool:
    """Determines if a PyTorch tensor has an integer dtype."""
    uint_types = [torch.uint8]
    sint_types = [torch.int8, torch.int16, torch.int32, torch.int64]
    if signed is None:
        return tensor.dtype in uint_types + sint_types
    elif signed:
        return tensor.dtype in sint_types
    else:
        return tensor.dtype in uint_types

class MyModel(nn.Module):
    def forward(self, x):
        if has_integer_dtype(x):
            return x + 1  # Example operation for integer tensors
        else:
            return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1, 3, 224, 224), dtype=torch.int32)

