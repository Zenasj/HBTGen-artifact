import torch.nn as nn

import os

import torch                                                                                                                                                    
import time
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None
try:
    from einops import rearrange
except ImportError:
    rearrange = None

causal = False

print("=====================TEST CUDNN Attention=====================")
b = 1
s = 4096
count = 0
torch.cuda.empty_cache()
while True:
    for h, d in zip((32, 16), (64, 128)):
        q, k, v = torch.randn(b, s, h*d*3, dtype=torch.bfloat16, device='cuda', requires_grad=True).chunk(3, dim=-1)
        q = q.view(b, -1, h, d).transpose(1, 2)
        k = k.view(b, -1, h, d).transpose(1, 2)
        v = v.view(b, -1, h, d).transpose(1, 2)
        with torch.no_grad():
            for i in range(5):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=causal)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
        print(f"{(b, s, h, d)} {t2-t1}")

    if b > 1:
        b //= 2
    s *= 2
    if s > 131072:
       break

print("=====================TEST Flash Attention=====================")
b = 1
s = 4096
count = 0
torch.cuda.empty_cache()
while True:
    for h, d in zip((32, 16), (64, 128)):
        q, k, v = torch.randn(b, s, h, 3*d, dtype=torch.bfloat16, device='cuda', requires_grad=True).chunk(3, dim=-1)
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
        cu_seqlens_k = cu_seqlens_q
        with torch.no_grad():
            for i in range(5):
                out = flash_attn_unpadded_func(
                        q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                        0.0, causal=causal
                    )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = flash_attn_unpadded_func(
                    q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                    0.0, causal=causal
                )
            torch.cuda.synchronize()
            t2 = time.perf_counter()
        print(f"{(b, s, h, d)} {t2-t1}")

    if b > 1:
        b //= 2
    s *= 2
    if s > 131072:
       break