# torch.rand(1, dtype=torch.float)  # Input shape is a single-element tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compute CPU clamp result
        cpu_clamped = x.cpu().clamp(0, 1)
        # Compute GPU clamp result (if available)
        if torch.cuda.is_available():
            gpu_clamped = x.cuda().clamp(0, 1).cpu()
        else:
            # If no GPU, assume discrepancy (since the issue is about CPU vs GPU)
            return torch.tensor(1.0)
        # Compare the two results. NaN != 1, so they differ in the issue's scenario
        return (cpu_clamped != gpu_clamped).float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([float('nan')], dtype=torch.float)

