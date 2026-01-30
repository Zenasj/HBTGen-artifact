import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature, _score_mod_signature, flex_attention
from torch._inductor.lowering import make_pointwise, register_lowering
from torch._inductor.virtualized import ops
from torch.nn.attention.flex_attention import create_block_mask

torch.set_default_device('cuda')

flex_attention = torch.compile(flex_attention, dynamic=False)

prefix_lengths = torch.arange(8)
def prefix_lm(b, h, q, kv):
    return prefix_lengths[b] >= kv

mask = create_block_mask(prefix_lm, 8, None, 512, 512, _compile=True)