import torch.nn as nn

import torch

d_model = 3
layer = torch.nn.TransformerEncoderLayer(d_model, 1, 6, batch_first=True)
layer.eval()
x = torch.randn(1, 5, d_model)
unmasked_output = layer(x)
is_causal_output = layer(x, is_causal=True)
mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
masked_output = layer(x, src_mask=mask)

assert not torch.equal(unmasked_output, is_causal_output)
assert torch.equal(masked_output, is_causal_output)