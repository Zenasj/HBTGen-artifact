import torch.nn as nn

import torch
import torch.export._trace
from torch._inductor.decomposition import decompositions, get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 64, 192, dtype=torch.float16))
        self.bias = torch.nn.Parameter(torch.randn(64, 1, 192, dtype=torch.float16))

    def forward(self, x):
        return torch.ops.aten.baddbmm.default(self.bias, x, self.weight)

x = torch.randn(64, 2048, 64, dtype=torch.float16, requires_grad=False)
inputs = (x,)

m = M()
gm = make_fx(m, pre_dispatch=False, decomposition_table=decompositions)(*inputs)
gm.print_readable(print_output=True)