import torch.nn as nn

import torch
import torch.nn.functional as F

print(torch.__version__)
x = torch.rand(1, 8, 5).to('mps')
x = x[..., :4]
print(x[..., 0])
atten = F.softmax(x, dim=1)
print(atten[..., 0])

x = x.to('cpu')
atten = F.softmax(x, dim=1)
print(atten[..., 0])

import torch

print(torch.__version__)

x = torch.rand(1, 8, 5).to('mps')
print(x[..., 0])

x = x.to('cpu')
print(x[..., 0])

tensor([[0.4302, 0.5965, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
       device='mps:0')
tensor([[0.4302, 0.5965, 0.0185, 0.1836, 0.2624, 0.7493, 0.2166, 0.9861]])