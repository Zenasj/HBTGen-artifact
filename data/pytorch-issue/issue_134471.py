import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import (
    BlockMask,
    _score_mod_signature,
)
import torch.nn.functional as F
from torch import nn


device = "cuda"
compile = True
contiguous = False


class Attn(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        bias=False,
    ):
        super().__init__()
        assert (
            dim % n_head == 0
        ), f"dim must be divisible by n_head found: {dim} and {n_head}"
        # key, query, value projections for all heads, but in a batch
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.n_head = n_head
        self.head_dim = dim // n_head
        self.n_embd = dim

    def forward(
        self, x, score_mod: None | _score_mod_signature, block_mask: None | BlockMask
    ):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        if contiguous:
            qkv = qkv.contiguous()
        q, k, v = qkv

        y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        return y


layer = Attn(512, 8).to(device)
if compile:
    layer.compile(mode="default")

x = torch.randn(2, 256, 512).to(device)
y = layer(x, score_mod=None, block_mask=None)