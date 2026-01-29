# torch.rand(B, C, H, W, dtype=...)  # The input shape is not explicitly defined in the issue, but it seems to be a 1D tensor.
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        v, inds = x.sort(descending=True)
        return inds

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input tensor in the issue is a 1D tensor with values ranging from 20 to 80.
    # For generalization, we can use a random 1D tensor with a similar range.
    return torch.rand(61) * 60 + 20

