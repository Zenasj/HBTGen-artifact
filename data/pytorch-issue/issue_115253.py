import torch.nn as nn

import torch
from torch.backends.cuda import sdp_kernel


def test_scaled_dot_product_fused_attention_vs_math_cpu(
    device, dtype, batch_size,
    seq_len, n_head, head_dim,
    causal,
):
    n_embd = n_head * head_dim
    x = torch.randn((batch_size, seq_len, 3 * n_head * head_dim), device=device, dtype=dtype, requires_grad=True)

    q, k, v = x.split(n_embd, dim=2)

    # (B, nh, T, hs)
    k = k.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
    q = q.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

    with sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
        actual = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
    actual.sum().backward()


test_scaled_dot_product_fused_attention_vs_math_cpu(
    device='cpu', dtype=torch.bfloat16,
    batch_size=12, seq_len=1030, n_head=3, head_dim=16, causal=True
)