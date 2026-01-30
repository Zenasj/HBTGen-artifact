import torch.nn as nn

b = torch.tensor([[0, 1]]).transpose(1, 0)
b.is_contiguous() #True

import torch
from torch.nn.functional import scaled_dot_product_attention

query = torch.tensor([[[[1, 2]]]], dtype=torch.float32)
query = query.transpose(-1, -2)
key = torch.tensor([[[[1]]]], dtype=torch.float32)
value = torch.tensor([[[[1]]]], dtype=torch.float32)

with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
    scaled_dot_product_attention(query, key, value)  # Works just fine

with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
    scaled_dot_product_attention(query, key, value)  # Fails, stacktrace below

new_a = torch.empty(a.shape).copy_(a)