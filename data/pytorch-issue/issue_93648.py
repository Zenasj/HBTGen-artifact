import torch.nn as nn

import torch
import torchdynamo
from torchdynamo.testing import rand_strided
from torchdynamo.debug_utils import run_fwd_maybe_bwd

torch._C._jit_set_nvfuser_single_node_mode(True)

from torch.nn import *
class ReproModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = Linear(196, 256)
        self.alpha = torch.randn(1, 1, 3, device='cuda')

    def forward(self, x : torch.Tensor):
        flatten_1 = torch.flatten(x, 2);
        transpose_2 = torch.transpose(flatten_1, 1, 2);

        # returns expected strided tensor like transpose_2
        # add_3 = torch.add(transpose_2, self.alpha)
        # returns contiguous tensor like alpha
        add_3 = torch.add(self.alpha, transpose_2)

        transpose_4 = torch.transpose(add_3, 1, 2);
        return self.linear_layer(transpose_4)

args = [((128, 3, 14, 14), (5376, 42, 14, 1), torch.float32, 'cuda', False)]
args = [rand_strided(sh, st, dt, de).requires_grad_(rg) for (sh, st, dt, de, rg) in args]

mod = ReproModule().cuda()
opt_mod = torchdynamo.optimize("aot_nvfuser")(mod)
res = run_fwd_maybe_bwd(opt_mod, args)