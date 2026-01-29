# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the requires_grad error when compiled with inductor
        return x[0].detach()  # Triggers the problematic code path in AOTAutograd

def my_model_function():
    # Returns the model instance causing the requires_grad error
    return MyModel()

def GetInput():
    # Returns input tensor with requires_grad=True to trigger the error
    return torch.rand(2, 2, requires_grad=True)

