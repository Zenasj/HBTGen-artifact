import torch
import torch.nn as nn

x = torch.randn(8, 16, 256)

# Works fine, returns Tensor of shape [8,16,256]
torch.nn.functional.scaled_dot_product_attention(x,x,x,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None)
# Triggers a TypeError
torch.nn.functional.scaled_dot_product_attention(x,x,x,None,0.0,False,None)