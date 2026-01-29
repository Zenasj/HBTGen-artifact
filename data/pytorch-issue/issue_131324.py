# torch.rand(B, H, dtype=torch.float32) for each tensor in the input tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x * y

def my_model_function():
    return MyModel()

def GetInput():
    B = 5  # Initial batch size (unbacked dimension)
    H = 6  # Dynamic dimension (marked as dynamic)
    x = torch.randn(B, H)
    y = torch.randn(B, H)
    # Apply Dynamo annotations as in the original repro
    torch._dynamo.decorators.mark_unbacked(x, 0)
    torch._dynamo.decorators.mark_unbacked(y, 0)
    torch._dynamo.mark_dynamic(x, 1)
    torch._dynamo.mark_dynamic(y, 1)
    return (x, y)

