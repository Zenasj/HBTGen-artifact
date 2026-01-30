import torch

x = torch.empty(2, 3, device="cuda")
mask = torch.ones_like(x).bool()
y = torch.tensor(0.3)  # cpu scalar tensor

def func(x, mask, y):
    return torch.masked_fill(x, mask, y)

from torch.fx.experimental.proxy_tensor import make_fx

from torch._decomp import get_decompositions
gm = make_fx(func, decomposition_table={})(x, mask, y)  # mimicing real use-case for TorchRefsNvfuserCapabilityMode context

from torch._prims.context import TorchRefsNvfuserCapabilityMode, TorchRefsMode
with TorchRefsNvfuserCapabilityMode():
    gm = make_fx(gm)(x, mask, y)
    #gm = make_fx(func)(x, mask, y)