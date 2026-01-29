# torch.rand(1100000000, 2, dtype=torch.float16, device='cuda:0', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.nn.functional.log_softmax(x, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1100000000, 2, dtype=torch.float16, device='cuda:0', requires_grad=True)

