import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex_attention = torch.compile(flex_attention)
create_block_mask = torch.compile(create_block_mask)

B = 1
S = 10_000
H = 8
D = 16

def sliding_window_mask(window_size: int):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
    return mask_fn
mask_mod = sliding_window_mask(1024)
block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device="cuda")

xs = [torch.randn(B, H, i, D, dtype=torch.float16, device="cuda", requires_grad=True) for i in range(10_000, 100_000, 10_000)]
for x in xs:
    out = flex_attention(x, x, x, block_mask=block_mask)
    bwd = out.sum().backward()

...
flex_attention = torch.compile(flex_attention, dynamic=True)
create_block_mask = torch.compile(create_block_mask, dynamic=True)

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
# from transformer_nuggets.utils.benchmark import profiler

flex_attention_compile = torch.compile(flex_attention, fullgraph=True)
fast_create = torch.compile(create_block_mask, fullgraph=True)

B = 1
S = 10_000
H = 8
D = 16

def sliding_window_mask(window_size: int):
    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= window_size // 2) & (kv_idx - q_idx <= window_size // 2)
    return mask_fn
mask_mod = sliding_window_mask(1024)

xs = [torch.randn(B, H, i, D, dtype=torch.float16, device="cuda", requires_grad=True) for i in range(10_000, 100_000, 1_000)]
# Warmpup
block_mask = fast_create(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device="cuda")
flex_attention_compile(xs[0], xs[0], xs[0], block_mask=block_mask)
from pathlib import Path
torch.cuda.synchronize()
# with profiler(Path("user_2.json")):
from contextlib import nullcontext
with nullcontext():
    for x in xs:
        block_mask = fast_create(mask_mod, B=None, H=None, Q_LEN=x.size(-2), KV_LEN=x.size(-2), device="cuda")
        out = flex_attention_compile(x, x, x, block_mask=block_mask)
        bwd = out.sum().backward()