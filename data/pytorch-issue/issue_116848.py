# torch.rand(2, 640, 32, 32, dtype=torch.float16, device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, hidden_states):
        return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 640, 32, 32, dtype=torch.float16, device='cuda')

