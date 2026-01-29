# torch.rand(1000000, dtype=torch.float32, device="cuda", requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, a):
        b = a * 2
        c = a * 2
        return b, c  # Returns two tensors that may be merged by Inductor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1000000, dtype=torch.float32, device="cuda", requires_grad=True)

