import torch
import torch.nn as nn

from torch._functorch.aot_autograd import aot_export_module
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        def true_fn(x):
            return x.cos()

        def false_fn(x):
            return x.sin()

        a = torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])
        return (a + 3, a + 4)

inp = torch.randn(3, 4)
gm, _ = aot_export_module(M(), (inp,), trace_joint=False)