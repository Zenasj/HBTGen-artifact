# torch.rand(B, T, C, dtype=torch.float32)
import torch
from torch.nn import Module, Linear
from torch.nn.attention.flex_attention import (
    flex_attention,
    BlockMask,
    _score_mod_signature,
)

class MyModel(Module):
    def __init__(self, dim, n_head, bias=False, contiguous=False):
        super().__init__()
        assert dim % n_head == 0, f"dim must be divisible by n_head found: {dim} and {n_head}"
        self.qkv = Linear(dim, 3 * dim, bias=bias)
        self.c_proj = Linear(dim, dim, bias=bias)
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.n_embd = dim
        self.contiguous_flag = contiguous  # renamed to avoid conflict with built-in function

    def forward(
        self, x, score_mod: None | _score_mod_signature = None, block_mask: None | BlockMask = None
    ):
        B, T, C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Rearranged to (3, B, n_head, T, head_dim)
        if self.contiguous_flag:
            qkv = qkv.contiguous()
        q, k, v = qkv  # Unpack along the first dimension (3 elements: q, k, v)
        y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        return y

def my_model_function():
    return MyModel(dim=512, n_head=8, contiguous=False)

def GetInput():
    return torch.randn(2, 256, 512, dtype=torch.float32).to('cuda')

