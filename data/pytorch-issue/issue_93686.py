# torch.rand(B, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.model2 = Model2()
    
    def forward(self, x):
        # Compare outputs of Model1 and Model2
        out1 = self.model1(x)
        out2 = self.model2(x)
        return torch.eq(out2, out1 * 2)  # Check if model2 output is twice model1's

class Model1(nn.Module):
    def forward(self, x):
        return torch.tensor(x.numel(), dtype=torch.int64)

class Model2(nn.Module):
    def forward(self, x):
        return torch.tensor(x.numel() + x.sin().numel(), dtype=torch.int64)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10)

