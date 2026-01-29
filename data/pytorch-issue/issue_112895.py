# torch.rand(320000, 768, dtype=torch.float32).cuda()
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.median(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(320000, 768, dtype=torch.float32).cuda()

