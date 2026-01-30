import torch.nn as nn

import torch
from torch import nn
import os

torch.manual_seed(12345)


class Foo(nn.Module):
    def __init__(self):
        super(Foo, self).__init__()
        self.bar = nn.Linear(3, 4)

    def forward(self, x):
        return self.bar(x)


if 'XXXX' in os.environ:
    foo = Foo()

x = torch.randn(2, 3)
print(x)