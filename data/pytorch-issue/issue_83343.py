# torch.rand(64, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        return F.pad(x, (0, 1), value=0.)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(64, device='mps' if torch.backends.mps.is_available() else 'cpu')

