import torch.nn as nn
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

batch_size = 2
num_heads = 32
head_dim = 128
num_tokens_q = 7
num_tokens_kv = num_tokens_q
device= "cuda"
dtype = torch.float16

num_pad_tokens = 3

query = torch.rand(batch_size, num_heads, num_tokens_q, head_dim, dtype=dtype, device=device) - 0.5
key = torch.rand(batch_size, num_heads, num_tokens_q, head_dim, dtype=dtype, device=device) - 0.5
value = torch.rand(batch_size, num_heads, num_tokens_q, head_dim, dtype=dtype, device=device) - 0.5


attn_mask_2d = torch.ones(batch_size, num_tokens_q, dtype=torch.int32, device=device)
attn_mask_2d[1][:num_pad_tokens] = 0  # simulate padding

attn_mask_4d = _prepare_4d_causal_attention_mask_for_sdpa(
    attn_mask_2d,
    input_shape=(batch_size, num_tokens_q),
    inputs_embeds=query,  # this is only used to retrieve device, dtype.
    past_key_values_length=0,
)

print("attn_mask_4d", attn_mask_4d)

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    sdpa_out_efficient = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask_4d
    )

with sdpa_kernel(SDPBackend.MATH):
    sdpa_out_math = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask_4d
    )

with sdpa_kernel(SDPBackend.MATH):
    sdpa_out_math_cpu = torch.nn.functional.scaled_dot_product_attention(
        query.cpu(),
        key.cpu(),
        value.cpu(),
        attn_mask=attn_mask_4d.cpu()
    )

print("[rocm math vs rocm mem-efficient] Median abs diff, non padded sequence:", (sdpa_out_efficient[0] - sdpa_out_math[0]).abs().median())
print("[rocm math vs rocm mem-efficient] Max abs diff, non padded sequence:", (sdpa_out_efficient[0] - sdpa_out_math[0]).abs().max())
print("[rocm math vs rocm mem-efficient] Median abs diff, padded sequence:", (sdpa_out_efficient[1, :, num_pad_tokens:] - sdpa_out_math[1, :, num_pad_tokens:]).abs().median())
print("[rocm math vs rocm mem-efficient] Max abs diff, padded sequence:", (sdpa_out_efficient[1, :, num_pad_tokens:] - sdpa_out_math[1, :, num_pad_tokens:]).abs().max())


sdpa_out_efficient = sdpa_out_efficient.cpu()
print("\n[cpu math vs rocm mem-efficient] Median abs diff, non padded sequence:", (sdpa_out_math_cpu[0] - sdpa_out_efficient[0]).abs().median())
print("[cpu math vs rocm mem-efficient] Max abs diff, non padded sequence:", (sdpa_out_math_cpu[0] - sdpa_out_efficient[0]).abs().max())
print("[cpu math vs rocm mem-efficient] Median abs diff, padded sequence:", (sdpa_out_math_cpu[1, :, num_pad_tokens:] - sdpa_out_efficient[1, :, num_pad_tokens:]).abs().median())
print("[cpu math vs rocm mem-efficient] Max abs diff, padded sequence:", (sdpa_out_math_cpu[1, :, num_pad_tokens:] - sdpa_out_efficient[1, :, num_pad_tokens:]).abs().max())

sdpa_out_math = sdpa_out_math.cpu()
print("\n[cpu math vs rocm math] Median abs diff, non padded sequence:", (sdpa_out_math_cpu[0] - sdpa_out_math[0]).abs().median())
print("[cpu math vs rocm math] Max abs diff, non padded sequence:", (sdpa_out_math_cpu[0] - sdpa_out_math[0]).abs().max())
print("[cpu math vs rocm math] Median abs diff, padded sequence:", (sdpa_out_math_cpu[1, :, num_pad_tokens:] - sdpa_out_math[1, :, num_pad_tokens:]).abs().median())
print("[cpu math vs rocm math] Max abs diff, padded sequence:", (sdpa_out_math_cpu[1, :, num_pad_tokens:] - sdpa_out_math[1, :, num_pad_tokens:]).abs().max())