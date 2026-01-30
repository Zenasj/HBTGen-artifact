import torch.nn as nn

import torch
from torch import nn

test = torch.ones(0, 2100).cuda()
f = nn.Linear(2100, 2100).cuda()
f(test).sum().backward()

import torch
from torch import nn

test = torch.ones(0, 2100)
f = nn.Linear(2100, 2100)
f(test).sum().backward()