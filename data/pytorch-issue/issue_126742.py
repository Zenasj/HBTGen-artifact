# torch.rand((), dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare behavior between dim=0 (current implementation) and expected error
        try:
            torch.aminmax(x, dim=0)  # Current implementation allows this
            return torch.tensor(1.0)  # 1 indicates the bug (dim case succeeded)
        except RuntimeError:
            return torch.tensor(0.0)  # 0 indicates correct behavior (dim case failed)

def my_model_function():
    # Returns the model instance for comparison testing
    return MyModel()

def GetInput():
    # Returns a 0D complex tensor to trigger the aminmax behavior
    return torch.tensor(7. + 5j, dtype=torch.complex64)

