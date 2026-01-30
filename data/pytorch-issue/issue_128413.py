import torch.nn as nn

import torch
from torch import nn


cache = []


# forward hook to save output
def hook(module, inputs, output):
    cache.append(output[0].detach())


enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True)
enc_layer.eval()

# register hook to get the output of the self-attention layer
handle = enc_layer.self_attn.register_forward_hook(hook)

# input tensor of shape (batch_size, seq_len, d_model)
x = torch.randn(4, 6, 32)

# forward pass
with torch.inference_mode():
    output = enc_layer(x)

# output of the self-attention layer
assert len(cache) == 1, f"Expected 1 output, got {len(cache)}"
print(cache[0].shape)

# remove hook
handle.remove()