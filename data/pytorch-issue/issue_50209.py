# torch.rand(4, dtype=torch.float32, device='cuda:0', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('indices', torch.tensor([0, 1], device='cuda:0'))  # Fixed indices as buffer

    def forward(self, x):
        return x.gather(0, self.indices)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32, device='cuda:0', requires_grad=True)

