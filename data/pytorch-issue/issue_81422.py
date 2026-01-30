import torch.nn as nn

import torch


class M(torch.nn.Module):
    def forward(self, x, y):
        print(y.shape)
        x = x.resize_(y.shape)
        return x, y


x = torch.tensor(1.2)
y = torch.tensor(4.2)

M()(x, y)
torch.jit.trace(M(), (x, y))