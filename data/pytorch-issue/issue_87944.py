# torch.randint(1, 6, (5,), dtype=torch.int64)  # shape (5,), dtype=torch.int64
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute(self, x):
        output = torch.log(torch.exp(-x**2))
        output = torch.atleast_1d(output)
        output = output.to(x.dtype)  # Cast to original dtype (e.g., int64)
        return output
    
    def forward(self, x):
        # Compute on current device
        current_out = self.compute(x)
        # Compute on the other device (CPU/CUDA toggle)
        other_device = "cuda" if x.is_cuda else "cpu"
        other_x = x.to(other_device)
        other_out = self.compute(other_x)
        # Bring tensors to same device for comparison
        if current_out.device != other_out.device:
            other_out = other_out.to(current_out.device)
        # Return True if outputs match across devices, else False
        return torch.all(current_out == other_out)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(1, 6, (5,), dtype=torch.int64)

