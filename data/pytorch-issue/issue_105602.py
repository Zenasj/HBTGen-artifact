import torch
import torch.nn as nn

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    torch.nn.functional.scaled_dot_product_attention(q,k,v)