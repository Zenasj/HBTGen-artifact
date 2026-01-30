import torch.nn as nn

import torch
from functorch.experimental.control_flow import cond


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, pred, x):
        def true_fn(val):
            return self.linear(val) * torch.tensor(2)

        def false_fn(val):
            return self.linear(val) * torch.tensor(-1)

        return cond(pred, true_fn, false_fn, [x])

mod = Module()
mod = torch.compile(mod)
x = torch.randn([3, 3])
pred = torch.tensor(x[0][0].item() < 0)
real_result = mod.forward(pred, x)

import torch
from functorch.experimental.control_flow import cond


x = torch.randn((3,))


def f1(x1, x2):
    return x1 + x2


def f2(x1, x2):
    return x1 * x2


@torch.compile()
def f(z):
    return cond(z, f1, f2, [x, x])

f(torch.tensor(True))