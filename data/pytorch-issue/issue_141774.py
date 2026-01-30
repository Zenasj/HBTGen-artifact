import torch.nn as nn

import torch
import torch.nn.functional as F

device = "mps"
# Test both bfloat16 and float16 for SDPA compatibility with autocast. See below output examples.
# dtype = torch.bfloat16
dtype = torch.float16

query = torch.rand(4, 4, 8, dtype=torch.float32, device=device)
key = torch.rand(4, 4, 8, dtype=torch.float32, device=device)
value = torch.rand(4, 4, 8, dtype=dtype, device=device)

with torch.amp.autocast(device_type=device):
    hidden_states = F.scaled_dot_product_attention(query, key, value)
# RuntimeError: Expected query, key, and value to have the same dtype, but got query.dtype: float key.dtype: float and value.dtype: c10::BFloat16 instead.
# RuntimeError: Expected query, key, and value to have the same dtype, but got query.dtype: float key.dtype: float and value.dtype: c10::Half instead.