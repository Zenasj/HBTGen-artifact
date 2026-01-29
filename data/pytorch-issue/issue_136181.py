# torch.rand(16, 16, 768, dtype=torch.bfloat16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        amax = torch.abs(torch.max(x))
        clamped = torch.clamp(amax, min=1e-12)
        scale = 448.0 / clamped
        return scale.to(torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    torch.manual_seed(0)  # Matches the repro's seed for consistency
    return torch.randn(16, 16, 768, dtype=torch.bfloat16, device="cuda")

