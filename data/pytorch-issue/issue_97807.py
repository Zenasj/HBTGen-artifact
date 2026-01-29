# torch.rand(B, C, H, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        a = F.pad(x, (0, 1))  # Pad last dimension (H) by (0,1)
        b = F.pad(a, (0, 0, 0, 1), 'reflect')  # Pad with reflect mode on penultimate dimension
        b[0, 0, 0] = 0.1  # In-place assignment causing potential compilation issues
        return b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 1, device='cuda', dtype=torch.float32)

