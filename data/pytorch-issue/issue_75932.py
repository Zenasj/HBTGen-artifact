# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cpp_ref = torch.zeros(2, 2)
        val = torch.zeros(2, 2)
        a = [None]
        b = [None, None]
        a[0] = b
        b[0] = a
        b[1] = val
        self.cpp_ref.grad = val
        del a, b, val
        # Explicitly trigger garbage collection to replicate the bug scenario
        import gc
        gc.collect()

    def forward(self, x):
        # Access the potentially deallocated tensor through the C++ reference
        return self.cpp_ref.grad  # May trigger undefined behavior due to dangling pointer

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input tensor (unused by model forward but required for interface compliance)
    return torch.rand(1, dtype=torch.float32)

