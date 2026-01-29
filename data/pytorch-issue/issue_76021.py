# torch.rand(2400, 128, dtype=torch.half, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize m2 with the sliced and transposed tensor as in the original example
        m2_initial = torch.rand(2048, 25272, dtype=torch.half, device='cuda').t()
        m2_sliced = m2_initial[21940:24340]  # Slices 2400 rows from the 25272-row tensor
        self.m2 = nn.Parameter(m2_sliced.t())  # Final shape: (2048, 2400)
        self.bias = nn.Parameter(torch.rand(128, dtype=torch.half, device='cuda'))
    
    def forward(self, x):
        return torch.addmm(self.bias, self.m2, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2400, 128, dtype=torch.half, device='cuda')

