# (torch.rand(B, C, W, dtype=torch.float, device='cuda'), torch.randint(0, C, (B, W), dtype=torch.long, device='cuda'))
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        x, t = inputs
        x0 = x[:, :, 0]  # Extract first slice along dim=2
        x3 = x[:, :, -1]  # Extract last slice along dim=2
        t0 = t[:, 0]
        t3 = t[:, -1]
        r0 = F.cross_entropy(x0, t0, reduction='none')
        r3 = F.cross_entropy(x3, t3, reduction='none')
        # Return 1.0 if outputs are close, 0.0 otherwise
        return torch.tensor([float(torch.allclose(r0, r3))], device=r0.device)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, W = 2, 10, 4  # Inferred from original issue's example
    x = torch.rand(B, C, W, dtype=torch.float, device='cuda')
    t = torch.randint(0, C, (B, W), dtype=torch.long, device='cuda')
    return (x, t)

