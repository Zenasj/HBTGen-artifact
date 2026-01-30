import torch.nn as nn

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend


query = torch.tensor([[[[1, 2]]]], dtype=torch.float32)
query = query.transpose(-1, -2).to("cuda")
key = torch.tensor([[[[1]]]], dtype=torch.float32).to("cuda")
value = torch.tensor([[[[1]]]], dtype=torch.float32).to("cuda")


with sdpa_kernel(SDPBackend.MATH):
    out_unpadded = scaled_dot_product_attention(query, key, value)  # Works fine

q_padding_amount = 4 - query.size(-1)
kv_padding_amount = 4 - key.size(-1)
query_padded = torch.nn.functional.pad(query, (0, q_padding_amount))
key_padded = torch.nn.functional.pad(key, (0, kv_padding_amount))
value_padded = torch.nn.functional.pad(value, (0, kv_padding_amount))

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    out_padded = scaled_dot_product_attention(query_padded, key_padded, value_padded)  # Fails, stacktrace below

out_sliced = out_unpadded[..., : query.size(-1)]

torch.testing.assert_close(out_unpadded, out_sliced)