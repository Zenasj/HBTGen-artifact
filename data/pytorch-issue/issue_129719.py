import torch.nn as nn

import torch
import os
from torch.fx.experimental.proxy_tensor import make_fx

class M(torch.nn.Module):
    def forward(self, x, y):
        return x+ y

inp = torch.rand(1)
inp2 = torch.rand(1)
args = (inp, inp2)

gm = make_fx(M(), tracing_mode="symbolic")(inp, inp2)
print(gm)
so = torch._inductor.aot_compile(gm, args)

gm = torch.export.export(M(), args).module()
print(gm)
so = torch._inductor.aot_compile(gm, args)