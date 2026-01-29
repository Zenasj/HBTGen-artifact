# torch.rand(0, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute sort on CPU
        cpu_sorted = x.sort()
        try:
            # Move input to MPS and compute sort
            mps_sorted = x.to('mps').sort()
            # Compare values and indices across devices
            values_ok = torch.allclose(cpu_sorted.values, mps_sorted.values.cpu())
            indices_ok = (cpu_sorted.indices == mps_sorted.indices.cpu()).all()
            return values_ok & indices_ok  # Returns torch.bool scalar
        except Exception:
            # Return False if MPS sort fails
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Empty 1D tensor matching the issue's scenario
    return torch.tensor([], dtype=torch.float32)

