import torch
import torch.nn as nn

class NativeMHA(nn.Module):
    """
    Modified from https://github.com/pytorch/pytorch/blob/v1.12.1/test/test_native_mha.py#L144-L166
    """

    def __init__(self, embed_dim, num_heads, qkv, proj):
        super().__init__()
        self.qkv = qkv
        self.proj = proj
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(
        self, q, k, v, key_padding_mask=None, need_weights=False, average_attn_weights=True
    ):
        return torch._native_multi_head_attention(
            q,
            k,
            v,
            self.embed_dim,
            self.num_heads,
            self.qkv.weight,
            self.qkv.bias,
            self.proj.weight,
            self.proj.bias,
            key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )

device = "cuda:0"
dtype = torch.float32
batch_size = 2
seq_length = 4
embed_dim = 16
num_heads = 4

qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
m = NativeMHA(embed_dim, num_heads, qkv, proj)
x = torch.rand(batch_size, seq_length, embed_dim, device=device, dtype=dtype)

m(x, x, x)  # everything is ok
with torch.autocast("cuda"):
    m(x, x, x)  # raise runtime error

# In my modules __init__
self.audio_self_attention = nn.MultiheadAttention(decoder_input_channels, num_heads=16, batch_first=True)