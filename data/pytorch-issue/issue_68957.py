# torch.rand(2, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input_shapes):
        # Extract dimensions from input tensor
        shape1_dim = input_shapes[0].item()
        shape2_dim = input_shapes[1].item()
        # Check if any dimension is negative (discrepancy condition)
        has_negative = (shape1_dim < 0) or (shape2_dim < 0)
        return torch.tensor(has_negative, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor encoding two shapes: [1] and [-12]
    return torch.tensor([1, -12], dtype=torch.int32)

