import torch.nn as nn

import os
import torch
import random
import numpy as np

class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=3):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.ones(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.ones(ensemble_size, in_features, out_features)))

    def forward(self, x):
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

#         I suppose the error is from x = x + bias, welcome to check it out
#         print(x)
#         print(bias)

        x = x + bias

        return x

device = torch.device("cpu")  # this is for producing results on cpu, in this case, you should comment out next line
# device = torch.device("mps")  # this is for producing results on mps, in this case, you should comment out previous line
model = EnsembleLinear(in_features=3, out_features=3, ensemble_size=3).to(device)
x = torch.arange(3*3, dtype=torch.float32).reshape(3,3).to(device)
y = model(x)
print(y)

# Here as you can see, y on cpu and mps are diffrent:
"""
y on cpu:
tensor([[[ 4.,  4.,  4.],
         [13., 13., 13.],
         [22., 22., 22.]],

        [[ 4.,  4.,  4.],
         [13., 13., 13.],
         [22., 22., 22.]],

        [[ 4.,  4.,  4.],
         [13., 13., 13.],
         [22., 22., 22.]]], grad_fn=<AddBackward0>)

y on mps:
tensor([[[ 4.0000,  4.0000,  4.0000],
         [ 4.0000,  4.0000,  4.0000],
         [ 4.0000,  4.0000,  4.0000]],

        [[13.0000, 13.0000, 13.0000],
         [13.0000, 13.0000, 13.0000],
         [13.0000, 13.0000, 13.0000]],

        [[22.0000, 22.0000, 22.0000],
         [22.0000, 22.0000, 22.0000],
         [22.0000, 22.0000, 22.0000]]], device='mps:0', grad_fn=<AddBackward0>)

"""