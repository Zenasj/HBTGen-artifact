import torch
import torch.nn as nn

class Bar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        def true_fn(x):
            return self.linear(x).cos()
        def false_fn(x):
            return self.linear(x).sin()
        return torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])

class CondExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bar = Bar()

    def forward(self, x):
        return x.cos() + self.bar(x)

inp = (torch.randn(4, 4),)
ep_strict = torch.export.export(CondExport(), inp)

gm_unflat_strict = ep_strict.module(flat=False)