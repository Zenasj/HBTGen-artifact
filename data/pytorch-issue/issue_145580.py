import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from torch.nn.attention import bias, sdpa_kernel, SDPBackend

@dataclass
class Config:
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 6
    n_ctx: int = 2048
    bias: bool = False

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)

        # HERE, WE NEED THIS CONTIGUOUS TO BE A NO-OP
        # y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = y.transpose(1, 2).view(B, T, C)
        y = self.c_proj(y)
        return y

def test_attention(backend: SDPBackend):
    config = Config()
    Attention = CausalSelfAttention(config).to("cuda", dtype=torch.float16)
    sample_input = torch.randn(1, 2048, config.n_embd, device="cuda", dtype = torch.float16)
    with sdpa_kernel(backend):
        try:
            out = Attention(sample_input)
            print("ALL GOOD")
        except RuntimeError as e:
            print("❗ NOT GOOD ❗")
            print(e)

if __name__ == "__main__":
    width = 100
    print("SDPA-Flash".center(width, "-"))
    test_attention(SDPBackend.FLASH_ATTENTION)
    print("SDPA-CuDNN".center(width, "-"))
    test_attention(SDPBackend.CUDNN_ATTENTION)