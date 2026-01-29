# torch.rand(10, 3, 5), torch.rand(10, 3, 4), torch.rand(10, 4, 5)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        M, batch1, batch2 = inputs
        return torch.baddbmm(M, batch1, batch2)

def my_model_function():
    return MyModel()

def GetInput():
    M = torch.rand(10, 3, 5)
    batch1 = torch.rand(10, 3, 4)
    batch2 = torch.rand(10, 4, 5)
    return (M, batch1, batch2)

