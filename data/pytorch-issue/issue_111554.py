# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x_view = x.view(-1)  # Create a view of x
        new_tensor = torch.ones(2, dtype=x.dtype, device=x.device)  # New tensor for set_()
        x.set_(new_tensor)  # Replace x's storage with new_tensor
        x_view.mul_(2)      # Mutate the original storage (now not referenced by x)
        return x            # Return the new tensor from set_()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2)  # Input tensor with 2 elements (shape (2,))

