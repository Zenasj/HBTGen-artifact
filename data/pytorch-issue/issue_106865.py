# torch.rand(0, 30, dtype=torch.float32), torch.rand(30, 40, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.mm(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(0, 30, dtype=torch.float32)
    y = torch.rand(30, 40, dtype=torch.float32)
    return (x.to("cuda"), y.to("cuda"))

