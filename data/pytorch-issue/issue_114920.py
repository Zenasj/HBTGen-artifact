# torch.rand(B, 1, 1, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Explicitly uses aten reshape with 'self' as a keyword argument
        return torch.ops.aten.reshape.default(self=x, shape=(2,))

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the required input shape (B=1, C=1, H=1, W=2)
    return torch.rand(1, 1, 1, 2, dtype=torch.float32)

