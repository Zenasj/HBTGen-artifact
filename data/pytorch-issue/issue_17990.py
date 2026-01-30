self.register_buffer('means', torch.zeros(1,nf,1,1))

self.means  = self.means.mul(self.mom).add(1-self.mom, m)

self.means *= self.mom
self.means += (1 - self.mom) * m

import torch
from torch import nn, optim
import torch.nn.functional as F

def l2norm(v, eps=1e-12):
    return v / (v.norm() + eps)

class OneIterationsSpectralNormMixin:
    def __init__(self, output_dim, eps=1e-12):
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        self.register_buffer("vector", torch.rand(1, output_dim))

    # Compute the spectrally-normalized weight
    def one_iteration_spectral_norm(self, weight):
        W = weight.view(weight.size(0), -1)
        # Apply one power iteration
        v = F.normalize(self.vector @ W, eps=self.eps)
        u = v @ W.t()
        # Compute this singular value
        sv = u.norm() + self.eps
        # If training, update the sv / vector to the new value
        if self.training:
            self.vector = u
        return weight / sv

class SpectralNormLinear(nn.Linear, OneIterationsSpectralNormMixin):
    def __init__(self, in_dim, out_dim, bias=True):
        nn.Linear.__init__(self, in_dim, out_dim, bias)
        OneIterationsSpectralNormMixin.__init__(self, out_dim)

    def forward(self, tensor):
        return F.linear(
            tensor, self.one_iteration_spectral_norm(self.weight), self.bias
        )

import torch
import torch.jit
import torch.nn as nn


class Mod(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('y', torch.zeros(1))

    def forward(self, x):
        self.y = x + self.y
        return self.y


j = torch.jit.trace_module(Mod(), dict(forward=torch.zeros(1)))

print(j(torch.ones(1)))
print(j(torch.ones(1)))
assert j.y == torch.ones(1) * 2, f'Expected value 2, got {j.y}'