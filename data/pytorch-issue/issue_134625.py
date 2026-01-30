import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention

torch.set_default_device("cuda")
torch.manual_seed(0)

flex_attention = torch.compile(flex_attention, dynamic=False)

B, H, T, D = 2, 8, 256, 32
# Create input tensors
query = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)
key = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)
value = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True)


# Forward pass
output = flex_attention(query, key, value)

# Compute loss and backward
loss = output.sum()
loss.backward()