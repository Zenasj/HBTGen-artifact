# torch.rand(2, 3, 4)  # First element's shape; input is a tuple of three tensors

import torch
from torch import nn
from typing import Tuple, Optional

class MyModel(nn.Module):
    def forward(self, input: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
        changed_input = input[0] + 1
        return (changed_input,) + input[1:]

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(2, 3, 4)
    b = torch.rand(2, 3, 4)
    c = torch.rand(2, 3, 4)
    return (a, b, c)

