import torch.nn as nn

import torch
import torch._dynamo
from torch import nn
import torch.nn.functional as F
import logging

torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True

class hsigmoid(nn.Module):

    def forward(self, x):
        out = F.relu6(x + 3)
        return out


class MyModule(nn.Module):

    def __init__(self, inp=4):
        super(MyModule, self).__init__()
        self.layers = nn.Sequential(nn.Linear(inp, inp, bias=False), hsigmoid())

    def forward(self, x):
        y = torch.rand([4, 4])
        z = self.layers(y).view(4, 4, 1, 1)
        return x * z.expand_as(x)

m = MyModule()
x = torch.rand([4, 4, 4, 4])
opt_m = torch._dynamo.optimize("inductor")(m)
print(m(x))
print(opt_m(x))