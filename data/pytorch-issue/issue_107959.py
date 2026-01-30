import torch.nn as nn

import torch
dtype = torch.bfloat16
seq_length = 8192
num_heads = 32
query_layer = torch.rand([1, num_heads, seq_length, 64], dtype=dtype, device="cuda:0")
key_layer = torch.rand([1, num_heads, seq_length, 64], dtype=dtype, device="cuda:0")
value_layer = torch.rand([1, num_heads, seq_length, 64], dtype=dtype, device="cuda:0")
alibi = torch.rand([1, num_heads, seq_length, seq_length], dtype=dtype, device="cuda:0")

# throw an error
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
     context_layer = torch.nn.functional.scaled_dot_product_attention(
         query_layer, key_layer, value_layer, attn_mask=alibi, dropout_p=0.0
     )

# no error
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
     context_layer = torch.nn.functional.scaled_dot_product_attention(
         query_layer, key_layer, value_layer, attn_mask=alibi, dropout_p=0.0
     )