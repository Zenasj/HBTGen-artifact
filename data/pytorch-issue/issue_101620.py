# torch.rand(1, 1, 5, 2, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        # Apply Gumbel-Softmax on the input tensor
        return F.gumbel_softmax(x, tau=0.15, hard=False, dim=-1)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 5, 2, dtype=torch.float32)

