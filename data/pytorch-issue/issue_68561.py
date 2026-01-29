# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Original approach (problematic)
        original = x.int()
        # Fixed approach using round()
        fixed = x.round().int()
        # Return both outputs and their equality check
        return original, fixed, torch.eq(original, fixed)

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a problematic input (float32) that triggers the issue with N=41
    N = 41.0
    return N * torch.tensor([1.0 / N], dtype=torch.float32)

