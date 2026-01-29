# torch.rand(10, 5, dtype=torch.float)
import torch
from torch import nn
from torch.distributions import Categorical

class MyModel(nn.Module):
    def forward(self, probs):
        return Categorical(probs).sample()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 5)

