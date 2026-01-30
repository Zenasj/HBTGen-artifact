import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.autograd.functional import jacobian
from torch.nn.utils import _stateless
from torch import nn
from torch.nn import functional as F

model = nn.Conv2d(3,1,1)
input = torch.rand(1, 3, 32, 32)
two_input = torch.cat([input, torch.rand(1, 3, 32, 32)], dim=0)
names = list(n for n, _ in model.named_parameters())

# This is exactly the same code as in issue #49171
jac1 = jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, input), tuple(model.parameters()))
jac2 = jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, two_input), tuple(model.parameters()))
assert torch.allclose(jac1[0][0], jac2[0][0])

class ResBasicBlock(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, (kernel_size, kernel_size), padding=kernel_size // 2,
                               bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, (kernel_size, kernel_size), padding=kernel_size // 2,
                               bias=False)
        self.norm1 = nn.BatchNorm2d(n_inner_channels)
        self.norm2 = nn.BatchNorm2d(n_channels)
        self.norm3 = nn.BatchNorm2d(n_channels)

    def forward(self, z, x=None):
        if x == None:
            x = torch.zeros_like(z)
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

model = ResBasicBlock(3, 1)
input = torch.rand(1, 3, 32, 32)
two_input = torch.cat([input, torch.rand(1, 3, 32, 32)], dim=0)
names = list(n for n, _ in model.named_parameters())

# This is exactly the same code as in issue #49171
jac1 = jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, input), tuple(model.parameters()))
jac2 = jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, two_input), tuple(model.parameters()))
assert torch.allclose(jac1[0][0], jac2[0][0])

model = ResBasicBlock(3, 1).double()
input = torch.rand(1, 3, 32, 32).double()
two_input = torch.cat([input, torch.rand(1, 3, 32, 32)], dim=0).double()