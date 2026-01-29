# torch.rand(32, dtype=torch.float, device='cuda'), torch.rand(32, dtype=torch.float, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs  # Unpack the input tuple
        if x.size(0) < 128:
            x = x * 2
        else:
            x = x * 3
        r = x + y
        r = r * y
        return r

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(32, dtype=torch.float, device='cuda')
    y = torch.rand(32, dtype=torch.float, device='cuda')
    return (x, y)  # Return tuple of two tensors

