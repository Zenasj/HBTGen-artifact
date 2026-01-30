import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(144).to('cuda')
        self.bn.training = False

    def forward(self, a, b):
        x1 = torch.cat([a, b], dim=1)
        x2 = self.bn(x1)
        return x2

a = torch.randn(1, 72, 112, 112, device='cuda')
b = torch.randn(1, 72, 112, 112, device='cuda')

m = torch.jit.trace(MyModule(), (a, b))
m.eval()
f = torch.jit.freeze(m)
f.eval()

f(a, b)
f(a, b)