# torch.rand(10, 10, dtype=torch.float32)  # Input shape is 2D matrix
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        inv_current = torch.linalg.inv(x)
        other_device = 'cpu' if x.is_cuda else 'cuda'
        try:
            x_other = x.to(other_device)
            inv_other = torch.linalg.inv(x_other)
            inv_other = inv_other.to(x.device)
            match = torch.allclose(inv_current, inv_other, atol=1e-5)
        except Exception:
            match = False  # Capture CUDA errors as mismatch
        return torch.tensor(match, dtype=torch.bool, device=x.device)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, dtype=torch.float32)

