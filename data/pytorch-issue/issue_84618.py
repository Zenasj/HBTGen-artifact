import torch.nn as nn

import torch
import torchdynamo
from torchdynamo.testing import rand_strided

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear = torch.nn.Linear(in_features=768, out_features=2304, bias=True)

    def forward(self, transpose_0, permute_1):
        add_2 = transpose_0 + permute_1
        layer_norm_3 = self.layer_norm(add_2)
        linear_4 = self.linear(layer_norm_3)
        return (linear_4,)

args = [((5, 784, 768), (602112, 1, 784), torch.float32, 'cuda', True),
        ((5, 784, 768), (602112, 1, 784), torch.float32, 'cuda', True)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]

mod = Repro().cuda()
opt_mod = torchdynamo.optimize("prims_nvfuser")(mod)
res = torchdynamo.debug_utils.run_fwd_maybe_bwd(opt_mod, args)