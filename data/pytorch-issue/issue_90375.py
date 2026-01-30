import torch.nn as nn

def random_masking(self, x, len_keep):
        N, L, D = x.shape  # batch, length, dim
        # len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample, ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is kept, 1 is masked
        mask = torch.ones([N, L], dtype=bool, device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

import os
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import functools
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch._dynamo.optimizations.backends import BACKENDS
from torch._dynamo.testing import rand_strided

# REPLACEABLE COMMENT FOR TESTING PURPOSES

args = [((384, 14, 14), (196, 14, 1), torch.int64, 'cuda', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_patch_embed_proj = Linear(in_features=32, out_features=768, bias=True).cuda()
        self.self_patch_embed_weight = torch.nn.Parameter(torch.randn([8192, 32], dtype=torch.float32)).cuda()



    def forward(self, x : torch.Tensor):
        self_patch_embed_weight = self.self_patch_embed_weight
        embedding = torch.nn.functional.embedding(x, self_patch_embed_weight);  x = self_patch_embed_weight = None
        self_patch_embed_proj = self.self_patch_embed_proj(embedding);  embedding = None
        reshape = torch.reshape(self_patch_embed_proj, (384, -1, 768));  self_patch_embed_proj = None
        _set_grad_enabled = torch._C._set_grad_enabled(False)
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        return (reshape,)


mod = Repro()

# Setup debug minifier compiler
torch._dynamo.debug_utils.MINIFIER_SPAWNED = True
compiler_fn = BACKENDS["dynamo_minifier_backend"]
raise RuntimeError(
    'Compiler name is None - this likely means that a custom compiler '
    'was called by torchdynamo. Please remove this error, import your '
    'custom compiler function, and replace the compiler_name="None" '
    'line below to compiler_name=<my_imported_custom_function>'
)

dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="None",
)
opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled=True):
    opt_mod(*args)