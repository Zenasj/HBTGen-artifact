import torch.nn as nn
import torch.nn.functional as F

import torch

class F(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, z):
        return x

    @staticmethod
    def backward(ctx, x):
        return x, x, x

op = F.apply


class M(torch.nn.Module):

    def forward(self, x, y, z):
        return op(x, y, z)

m = M()

def hook(m, i, o):
    return i[0].svd()

m.register_backward_hook(hook)
a = torch.randn(5, 5, requires_grad=True)
b = torch.randn(5, requires_grad=True)
c = torch.randn(5, 5, requires_grad=True)
m(a, b, c).sum().backward()