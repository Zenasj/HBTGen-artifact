import torch

x = torch.empty(1024, 1, device="cuda")
target = torch.empty(1024, device="cuda").long()

def func(x, target, index):
    nll_loss_forward = torch.ops.aten.nll_loss_forward.default(x, target, None, 1, index)

from torch.fx.experimental.proxy_tensor import make_fx

from torch._decomp import get_decompositions
decomp_table = get_decompositions({torch.ops.aten.nll_loss_forward})
gm = make_fx(func, decomposition_table=decomp_table)(x, target, 512)  # tracing with decomposition works
print(gm.graph)

from torch._prims.context import TorchRefsMode
with TorchRefsMode():
    gm = make_fx(func)(x, target, 512)  # somehow tracing here with decomposition breaks.
    print(gm.graph)

py
import torch
from torch._prims.context import TorchRefsMode
a = torch.randn(3, 3, device='cuda')
b = torch.randn(3, 3, device='cuda')

def func(a, b):
    return a.to(b)
with TorchRefsMode():
    func(a, b)