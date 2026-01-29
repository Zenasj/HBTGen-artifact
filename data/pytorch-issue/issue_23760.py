# torch.randint(0, 255, (256, 900, 2), dtype=torch.long, device='cuda') ← inferred input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 255, (256, 900, 2), dtype=torch.long, device='cuda')

