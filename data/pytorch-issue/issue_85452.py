import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._prims.context import TorchRefsMode
a = torch.randn(3, 3)
def f(a):
    return a.permute(1, 0)
with TorchRefsMode():
    gm = make_fx(f)(a)

x = torch.randn(2, 3, 4).permute(1, 2, 0)
torch.permute(x, (1, 2, 0))  # this works
# torch.permute(x, 1, 2, 0)  # this breaks!