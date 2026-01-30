import torch.nn as nn

import torch
from torch import nn

net = nn.Sequential(nn.BatchNorm1d(4)).cuda()

o = torch.randn(2, 4).cuda()

# auto-cast should automatically cast stuff
with torch.cuda.amp.autocast():
    for layer in net:
        o = layer(o)
        print(o.dtype)

import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 5)).cuda()
o = torch.randn(2, 4).cuda()

# auto-cast should automatically cast stuff
with torch.cuda.amp.autocast():
    for layer in net:
        o = layer(o)
        print(o.dtype)