# torch.randint(-100, 100, (20, 20), dtype=torch.int16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.argmax(0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(-100, 100, (20, 20), dtype=torch.int16, device='cuda')

