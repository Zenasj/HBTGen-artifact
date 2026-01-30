import torch.nn as nn

import torch
import code

bs = 1
nt_len = 1   # 5

q = torch.nested.nested_tensor( [torch.rand( (bs, 8, 16), dtype=torch.float16, device='cuda') for _ in range(nt_len) ], layout=torch.jagged)
k = torch.nested.nested_tensor( [torch.rand( (bs, 8, 16), dtype=torch.float16, device='cuda') for _ in range(nt_len) ], layout=torch.jagged)
v = torch.nested.nested_tensor( [torch.rand( (bs, 8, 16), dtype=torch.float16, device='cuda') for _ in range(nt_len) ], layout=torch.jagged)

q = q.transpose( 2, 1)
k = k.transpose( 2, 1)
v = v.transpose( 2, 1)

att = torch.nn.functional.scaled_dot_product_attention
with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
  out = att( q, k, v).transpose( 2, 1)

print( out.shape)