# torch.rand(B, 2, dtype=torch.float, device='cuda')  # Inferred input shape: (n_rows, 2)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.b = nn.Parameter(torch.ones(2, 2, device='cuda').float())  # Matches the original 'b' in the test

    def forward(self, a):
        return torch.mm(a, self.b)

def my_model_function():
    return MyModel()

def GetInput():
    n_rows = 0b01000000000000000000001  # 2^21 + 1 (one of the problematic input sizes)
    return torch.ones(n_rows, 2, dtype=torch.float, device='cuda')  # Uses the same input pattern as the issue's test

