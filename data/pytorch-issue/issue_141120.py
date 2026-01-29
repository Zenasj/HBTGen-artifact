# torch.rand(1, 1, 1, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Register fallthrough kernel for mul.Tensor on CPU to trigger the error
        lib = torch.library.Library("aten", "IMPL", "CPU")
        lib.impl(torch.ops.aten.mul.Tensor, torch.library.fallthrough_kernel)

    def forward(self, x):
        return 5 * x  # This operation uses the registered fallthrough kernel

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 3)

