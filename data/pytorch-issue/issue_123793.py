# torch.randint(1, 10, (2,), dtype=torch.int64, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        device = x.device
        s, s2 = x.tolist()
        g = torch.randn(s, device=device)
        g2 = torch.randn(s2, device=device)
        return torch.cat([g, g, g2])

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(1, 10, (2,), dtype=torch.int64, device='cuda')

