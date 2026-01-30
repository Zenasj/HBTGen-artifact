import torch.nn as nn

import torch
import torch.nn.functional as F
print(torch.__version__)
# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
attn_mask_bool = torch.randint(low=0,high=2,size=(32,8,128,128), device='cuda') > 0
attn_mask_float = torch.zeros((32,8,128,128), device='cuda', dtype=torch.float16)
attn_mask_float.masked_fill_(attn_mask_bool.logical_not(), float("-inf"))

res = torch.allclose(
    F.scaled_dot_product_attention(query,key,value, attn_mask=attn_mask_bool),
    F.scaled_dot_product_attention(query,key,value, attn_mask=attn_mask_float)
)
print(res) # returns True