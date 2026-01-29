# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        view_out = x.view(-1)
        reshape_out = x.reshape(-1)
        # Return both outputs to ensure both paths are part of the computation graph
        return (view_out, reshape_out)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32)

