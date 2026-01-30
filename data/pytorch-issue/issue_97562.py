import torch.nn as nn

def forward(self, xq : torch.Tensor, freqs_cis : torch.Tensor):
        view_as_complex = torch.view_as_complex(xq);  xq = None
        mul = view_as_complex * freqs_cis;  view_as_complex = freqs_cis = None
        return (mul,)

import torch
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

args = [
    ((128, 4, 12, 32, 2), (3072, 768, 64, 2, 1), torch.float32, "cuda", True),
    ((128, 1, 1, 32), (32, 32, 32, 1), torch.float32, "cuda", False),
]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


class RotaryLLAMA(torch.nn.Module):
    """Facebook implementation of rotary embeddings, simplified"""

    def __init__(self):
        super().__init__()

    def forward(self, xq: torch.Tensor, freqs_cis: torch.Tensor):
        xq_ = torch.view_as_complex(xq)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq)


mod = RotaryLLAMA()
opt_mod = torch._dynamo.optimize("inductor")(mod)

ref = run_fwd_maybe_bwd(mod, args)
res = run_fwd_maybe_bwd(opt_mod, args)

import torch
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

args = [
    torch.randn(128, 4, 12, 64, device=torch.device("cuda:0"), requires_grad=True),
    torch.randn(128, 1, 1, 32, device=torch.device("cuda:0"), dtype=torch.complex64, requires_grad=False),
]


class RotaryLLAMA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xq, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq)


mod = RotaryLLAMA()
opt_mod = torch._dynamo.optimize("inductor")(mod)

ref = run_fwd_maybe_bwd(mod, args)
res = run_fwd_maybe_bwd(opt_mod, args)