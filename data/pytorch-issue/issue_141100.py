# torch.rand(5, 4, 4, 1, 3, dtype=torch.double)  # Input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Output size set to (5,1,3) to match the reported gradient shape (5,4,5,1,3)
        self.pool = nn.AdaptiveMaxPool3d(output_size=(5, 1, 3))

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Requires_grad enabled to trigger backward pass and expose the bug
    return torch.rand(5, 4, 4, 1, 3, dtype=torch.double, requires_grad=True)

