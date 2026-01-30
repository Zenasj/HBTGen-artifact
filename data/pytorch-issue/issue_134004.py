import torch.nn as nn

import torch
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask, BlockMask, _score_mod_signature
import torch.nn.functional as F
import math
from torch import nn
from tensordict import TensorDict

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float = 0.0,
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
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.n_embd = dim
        self.dropout = dropout

    def forward(
        self, x, score_mod: None | _score_mod_signature, block_mask: None | BlockMask
    ):
        B, T, C = (
            x.size()
        )  


        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv
   
        y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        
        y.mean().backward() 
        
  
def noop(score, b, h, q_idx, kv_idx):
    return score


device = "cuda"
attn_layer = SelfAttentionLayer(512, 8).to(device)
x = torch.randn(2, 256, 512).to(device)
out = attn_layer(x,score_mod=noop,block_mask=None)