import torch.nn as nn

import torch
from torch import nn


class Module(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def mul_100(self, x):
        return x * 100

    def add_100(self, x):
        return x + 100


x = torch.ones(1)
module = torch.jit.trace_module(Module(),
                                {'forward': x, 'mul_100': x, 'add_100': x})
# print False
print(all(hasattr(module, method_name)
          for method_name in ['forward', 'mul_100', 'add_100']))