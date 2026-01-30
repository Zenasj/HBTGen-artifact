import torch
import torch.nn as nn

class Module(torch.nn.Module):
    def forward(self, pred, a):
        def true_fn(x):
            return torch.zeros_like(x)

        def false_fn(x):
            return torch.ones_like(x)

        b = torch.ones_like(a)
        c = torch.cond(pred, true_fn, false_fn, [b])
        return c

torch.compile(Module())(
    torch.tensor(True),
    torch.rand(10, 20),
)