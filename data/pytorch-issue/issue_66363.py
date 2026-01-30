import torch
import torch.nn as nn
# from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm


class Tt(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1)))

    def forward(self, x):
        x = self.f(x)
        x = self.f(x) # here used twice
        return x


a = torch.rand(4, 3, 24, 24)
model = Tt()
res = model(a)
loss = torch.mean(torch.abs(res - a))
loss.backward()