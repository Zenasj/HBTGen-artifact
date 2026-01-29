# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(3)  # Using BatchNorm1d instead of LazyBatchNorm1d to avoid the INTERNAL ASSERT error

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C = 2, 3
    return torch.randn(B, C, requires_grad=True)

