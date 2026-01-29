# torch.rand(5, dtype=torch.float32)  # Input is a 1D tensor of size 5
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        out, _ = torch.topk(x, 2)
        return out.mean()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, dtype=torch.float32, device='cuda', requires_grad=True)

