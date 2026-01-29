# torch.rand(1, dtype=torch.complex128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        finfo = torch.finfo(torch.complex128)
        val = finfo.min / 2
        self.nom = nn.Parameter(torch.tensor([complex(val, val)], dtype=torch.complex128))
        self.denom = nn.Parameter(torch.tensor([complex(val, val)], dtype=torch.complex128))
    
    def forward(self, x):
        return self.nom / self.denom

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.complex128)

