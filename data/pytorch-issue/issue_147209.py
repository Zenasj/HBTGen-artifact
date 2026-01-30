import torch
import torch.nn as nn

batch_size = 1
context_length = 4
embed_dim = 4
nheads=1

model = nn.TransformerEncoderLayer(embed_dim, nheads, dropout = 0, batch_first = True, norm_first = True)

input_tensor = torch.randn(batch_size, context_length, embed_dim)
padding_mask = torch.randint(low=0, high = 2, size = (batch_size, context_length))


bool_output = model(input_tensor, src_key_padding_mask = padding_mask.to(torch.float32))
float_output = model(input_tensor, src_key_padding_mask = padding_mask.to(torch.bool))

print(torch.allclose(bool_output, float_output, equal_nan = True, atol=1e-05))