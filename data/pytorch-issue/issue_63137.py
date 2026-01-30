import torch.nn as nn

import torch
import torch.fx

class Net(torch.nn.Module):
    def forward(self, x):
        return x @ x

Net()(torch.rand(4,4)) # OK
net = torch.fx.symbolic_trace(Net()) # error: unsupported operand type(s) for @: 'Proxy' and 'Proxy'