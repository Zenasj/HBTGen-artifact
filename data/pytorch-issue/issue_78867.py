# torch.rand(5, 5, 5, dtype=torch.float64, device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Target shape (5) as in original issue's example
        self.target = nn.Parameter(torch.rand(5, dtype=torch.float64, device='cuda', requires_grad=True))
    
    def forward(self, input):
        # Reproduces the kl_div scenario with shape mismatch
        return F.kl_div(input, self.target)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (5,5,5) and CUDA device from the issue's example
    return torch.rand(5, 5, 5, dtype=torch.float64, device='cuda', requires_grad=True)

