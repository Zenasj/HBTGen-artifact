# torch.rand(4, 4, 4, 4, 4, 4, dtype=torch.float32, device='cuda')  # Input x shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs  # Unpack the tuple input
        mean = torch.mean(x, dim=[2, 3, 4, 5], keepdim=True)
        return mean + y

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(4, 4, 4, 4, 4, 4, device='cuda')
    y = torch.rand((), device='cuda')
    return (x, y)  # Returns a tuple of inputs for the model

