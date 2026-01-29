# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class TensorLike:
    def __torch_function__(self, func, types, args=(), kwargs=None):
        print(f"__torch_function__ called for {func.__name__}")
        # Return a dummy tensor to mimic successful dispatch
        return torch.tensor([0.0])

class MyModel(nn.Module):
    def forward(self, x):
        # Create TensorLike instances and trigger torch.cat
        a = TensorLike()
        b = TensorLike()
        return torch.cat([a, b])

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a dummy tensor (not used in forward)
    return torch.rand(1)

