import torch.nn as nn

import torch
from torch.nn.utils.rnn import pack_padded_sequence
#bs, max_sequence_len, emb_dim
x = torch.zeros([16,10,20])
length = torch.zeros([16])
mask = [length != 0]
masked_x = x[mask]
masked_length = length[mask]
print(masked_x.shape, masked_length.shape)
pack_padded_sequence(masked_x, masked_length, batch_first = True)

TORCH_CHECK(input.dim() > 1, "The input must have at least 2 dimensions.");