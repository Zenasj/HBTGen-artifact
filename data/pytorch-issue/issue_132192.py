# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return x.cos()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Create a nested tensor with the specified structure
    data = torch.randn(10, 10)
    offsets = torch.tensor([0, 1, 3, 6, 10])
    nt = torch.nested.nested_tensor_from_jagged(data, offsets)
    return nt

