# torch.rand(B, S, D, dtype=torch.float32, device='cuda')  # Inferred from issue's example (10,5,16) on CUDA

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # The error occurs when converting CUDA tensors to nested due to device mismatch in offsets/values
        return torch.nested.as_nested_tensor(x, layout=torch.jagged)

def my_model_function():
    # Returns the model instance that triggers the bug
    return MyModel()

def GetInput():
    # Returns a CUDA tensor that causes the AssertionError in the original bug report
    return torch.rand(10, 5, 16, dtype=torch.float32, device='cuda')

