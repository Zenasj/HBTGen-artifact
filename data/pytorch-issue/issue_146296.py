import torch.nn as nn

import torch
from torch import nn


# UserError: Could not guard on data-dependent expression Eq(507 - u0, 0) (unhinted: Eq(507 - u0, 0)).  (Size-like symbols: u0)
class Repro(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cache, update, pos):
        _, _, max_seq_len, _ = cache.shape
        _, _, seqlen, _ = update.shape

        pos_item = pos[0].item() # u0
        torch._check(pos_item + seqlen <= max_seq_len) # u0 + 502 <= 507
        torch._check(pos_item >= 0)
        before = cache.narrow(2, 0, pos_item)

        # FAIL
        # Laith: why can't we make unbacked expressions size-like?
        after = cache.narrow(2, (pos_item + seqlen), (max_seq_len - pos_item - seqlen))

        # PASS
        end = torch.tensor(max_seq_len - pos_item - seqlen).item()
        after = cache.narrow(2, (pos_item + seqlen), end)

        return torch.cat([before, update, after], dim=2)


repro = Repro()

bsz = 1
n_heads = 4
max_seq_len = 512
head_dim = 64
seqlen = 5
pos_item = 1

cache = torch.zeros(bsz, n_heads, max_seq_len, head_dim)
update = torch.ones(bsz, n_heads, seqlen, head_dim)
pos = torch.tensor([pos_item])
example_inputs = (cache, update, pos)


torch.export.export(repro, example_inputs)