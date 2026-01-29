# torch.rand(1, 2, 2, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare view, reshape, and resize behaviors
        view_out = x.view(-1)
        reshape_out = x.reshape(-1)
        # Clone to avoid in-place modification and test non-destructive resize
        resized_clone = x.clone()
        resized_clone.resize_(x.numel())  # Workaround for resize(-1)
        resize_out = resized_clone.view(-1)  # Ensure 1D for comparison
        
        # Check if view and reshape outputs match
        view_reshape_close = torch.allclose(view_out, reshape_out, atol=1e-5)
        # Check if resize output matches view (fails for non-contiguous cases)
        view_resize_close = torch.allclose(view_out, resize_out, atol=1e-5)
        
        return torch.tensor([view_reshape_close, view_resize_close], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 2, 1, dtype=torch.float32)

