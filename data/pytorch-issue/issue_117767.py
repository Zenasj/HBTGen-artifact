# torch.rand(1, dtype=torch.float32, device="cuda")  # Dummy input tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.rand(2, 3, device="cuda"))
        self.param.grad = torch.rand_like(self.param)
        self.optim = torch.optim.AdamW([self.param])
    
    def forward(self, x):
        self.optim.step()
        return x  # Dummy return to satisfy forward contract

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, device="cuda")  # Dummy input tensor

