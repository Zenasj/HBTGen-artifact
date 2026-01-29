# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn
from torch.distributions import Normal

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        normal = Normal(inputs, 1.0)
        return normal.log_prob(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

