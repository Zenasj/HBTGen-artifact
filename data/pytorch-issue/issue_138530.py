# torch.rand(5, 4, 4, dtype=torch.float32, device='cuda:0')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute eigenvalues using batched computation
        batch_eigen = torch.linalg.eigvalsh(x)
        # Compute eigenvalues using per-matrix loop
        loop_eigen = torch.stack([torch.linalg.eigvalsh(m) for m in x])
        # Return absolute difference between results
        return torch.abs(batch_eigen - loop_eigen)

def my_model_function():
    return MyModel()

def GetInput():
    # Match input shape and device from original issue's example
    return torch.rand(5, 4, 4, dtype=torch.float32, device='cuda:0')

