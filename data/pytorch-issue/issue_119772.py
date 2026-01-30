import torch.nn as nn

import math
import torch
from torch.nn.functional import scaled_dot_product_attention

def sdp_attention(q, k, v):
    """
    q shape: (B, H, L, D)
    """
    with torch.cuda.amp.autocast(enabled=True, dtype=q.dtype):
        scale = 1 / math.sqrt(q.size(-1))
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        result = attn @ v
    return result

B = 5
H = 4
L = 10
D = 8
q = torch.randn((B, H, L, D)).to('cuda')
k = torch.randn((B, H, L, D)).to('cuda')
v = torch.randn((B, H, L, D)).to('cuda')

r1_fp32 = scaled_dot_product_attention(q, k, v)
r1_fp16 = scaled_dot_product_attention(q.half(), k.half(), v.half())
r1_bf16 = scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
print(f"max diff r1_fp32 vs r1_fp16 = {(r1_fp32 - r1_fp16).abs().max()}")
print(f"max diff r1_fp32 vs r1_bf16 = {(r1_fp32 - r1_bf16).abs().max()}")

r2_fp32 = sdp_attention(q, k, v)
r2_fp16 = sdp_attention(q.half(), k.half(), v.half())
r2_bf16 = sdp_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
print(f"max diff r1_fp32 vs r2_fp32 = {(r1_fp32 - r2_fp32).abs().max()}")
print(f"max diff r1_fp32 vs r2_fp16 = {(r1_fp32 - r2_fp16).abs().max()}")
print(f"max diff r1_fp32 vs r2_bf16 = {(r1_fp32 - r2_bf16).abs().max()}")