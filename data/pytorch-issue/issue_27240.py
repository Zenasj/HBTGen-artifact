import torch
import torch.nn

x = torch.randn(128, 30754)
fc1 = nn.Linear(30754, 512)
y = fc1(x)

"""
OUTPUT:
y is a tensor of shape (128, 512) that is filled with nan:
tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], grad_fn=<AddmmBackward>)
"""

fc2 = nn.Linear(30754, 1)
z = fc2(x) # z is a (128, 1) real-valued vector

import torch
import torch.nn as nn
x = torch.randn(128, 30754)
fc1 = nn.Linear(30754, 512)
y = fc1(x)

import torch
import torch.nn as nn
x = torch.randn(128, 30754, device='cuda')
fc1 = nn.Linear(30754, 512).cuda()
y = fc1(x)