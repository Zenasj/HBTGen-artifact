# (torch.rand(1, device='cuda'), torch.rand(1, device='cuda'))  # Input is a tuple of two tensors of shape (1,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pred = torch.tensor(True, device='cuda')  # Fixed predicate from the issue example

    def forward(self, inputs):
        a, b = inputs
        # Direct use of torch.add and torch.mul in torch.cond to reproduce the error
        return torch.cond(self.pred, torch.add, torch.mul, [a, b])

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(1, device='cuda')
    b = torch.rand(1, device='cuda')
    return (a, b)

