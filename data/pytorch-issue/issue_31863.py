import torch.nn as nn
import torch.nn.functional as F

py
import torch as T
from torch import nn
from torch.nn import functional as F

class Foo(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Linear(din, dout)
    def forward(self, x):
        y0, y1, _ = self.w(x).chunk(3, -1)
        return y0, y1


x = T.randn(100, 100)
foo = Foo(100, 100)
foo = T.jit.script(foo)
y0, y1 = foo(x)
loss = F.mse_loss(y0, y1)