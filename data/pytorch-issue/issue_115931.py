# torch.rand(2, 10), torch.rand(1, 10)  # Input shapes for xs and x respectively

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        xs, x = inputs
        indices = torch.tensor([0], device=xs.device)  # Ensure 1D index tensor
        new_xs = xs.clone()  # Avoid in-place modification for stability
        new_xs.index_copy_(0, indices, x)
        return new_xs

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device('cpu')
    xs = torch.ones(2, 10, device=device)  # Original xs shape [2,10]
    x = torch.zeros(1, 10, device=device)  # Original x shape [1,10]
    return (xs, x)

