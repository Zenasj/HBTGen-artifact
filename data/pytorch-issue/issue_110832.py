import torch.nn as nn

import torch
from torch import nn

class BadModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.l = nn.Linear(8, 16, bias=False)

  def forward(self, x):
    return torch.nn.functional.scaled_dot_product_attention(x, x, self.l(x), is_causal=True)

m = torch.compile(BadModule().to('cuda'))
x = torch.rand(1, 2, 4, 8).to('cuda') # (batch, num_heads, seq_len, embed_dim)
y = m(x) # forward pass
z = y.mean().backward() # fake backward pass