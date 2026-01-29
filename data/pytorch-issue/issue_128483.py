# Input: (torch.randn(1), torch.tensor(0, dtype=torch.int64, device='cuda'))  # Tuple of (x, index)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, args):
        x, idx = args
        return torch.gather(x, 0, idx)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(1, requires_grad=False).cuda()
    idx = torch.tensor(0, dtype=torch.int64, device='cuda')
    return (x, idx)

