# torch.rand(4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.distributions.Categorical(probs=x).entropy()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, dtype=torch.float32)

