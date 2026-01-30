import torch.nn as nn

import torch
import numpy as np

embedding_dim = 8
batch_size = 1
num_heads = 2
seq_len = 4

net = torch.nn.MultiheadAttention(embedding_dim, num_heads, add_zero_attn=True)
mask = torch.from_numpy(np.array([[True, True, True, True],
                                  [False, True, True, True],
                                  [False, False, True, True],
                                  [False, False, False, True]])).float() * -10000.0

for i in range(seq_len):
    x = torch.ones(seq_len, batch_size, embedding_dim, requires_grad=True)
    o, w = net(x, x, x, attn_mask=mask)
    # o.shape is (seq_len, batch_size, embedding_dim)
    o.mean([1, 2])[i].backward()
    print(i, x.grad.sum([1, 2]).view(-1))

# expected output:
# 0, [0, 0, 0, 0] # it means that the first output(first token in the sequence) does not depend on any input
# 1, [x, 0, 0, 0] # it means that the second output only depends on the first token
# 2, [y, z, 0, 0]
# 3, [t, w, u, 0]

# observed output:
# 0, [0, 0, 0, 0] # this one seems right
# 1, [x, y, 0, 0] # this is obviously wrong(the provided mask should have prevented the second token from attending to itself)
# 2, [z, t, w, 0]
# 3, [u, a, b, c]

attn_output_weights.masked_fill_(attn_mask, float('-inf'))
attn_output_weights = softmax(attn_output_weights, dim=-1)

attn_output_weights = softmax(attn_output_weights, dim=-1)
attn_output_weights.masked_fill_(attn_mask, 0.0)

attn_output_weights.masked_fill_(attn_mask, float('-inf'))
attn_output_weights = softmax(attn_output_weights, dim=-1)
attn_output_weights.masked_fill_(attn_mask, 0.0)