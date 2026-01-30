import torch.nn as nn

import torch
from functorch import make_fx
from einops import rearrange

# torch.fx.wrap('rearrange')

class B(torch.nn.Module):
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        return x

b = B()
tensor = torch.zeros([2, 2, 3, 2])

b2 = make_fx(
    b,
    decomposition_table={},
    # tracing_mode="symbolic",
    _allow_non_fake_inputs=True,
    _allow_fake_constant=False,
)(tensor)

b3 = make_fx(
    b,
    decomposition_table={},
    tracing_mode="symbolic",
    _allow_non_fake_inputs=True,
    _allow_fake_constant=False,
)(tensor)