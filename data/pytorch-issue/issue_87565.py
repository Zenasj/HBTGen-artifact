# torch.rand(5, 5, dtype=torch.float32)  # Inferred input shape (5,5)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute condition number on CPU and CUDA
        x_cpu = x.to('cpu')
        cond_cpu = torch.linalg.cond(x_cpu)
        
        # Check CUDA availability (required for the issue's comparison)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: cannot compare CPU/CUDA behavior")
        x_cuda = x.to('cuda')
        cond_cuda = torch.linalg.cond(x_cuda)
        
        # Compare results using a tolerance that accounts for large finite numbers
        are_close = torch.allclose(cond_cpu, cond_cuda, atol=1e16)
        return torch.tensor(not are_close, dtype=torch.bool)  # Return True if different

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(5, 5, dtype=torch.float32)

