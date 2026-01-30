import torch.nn as nn
import torch.nn.functional as F
import random

3
np.random.seed(12)
in1 = torch.FloatTensor(np.random.rand(256**2, 1, 64))
in2 = torch.FloatTensor(np.random.rand(128, 1, 64))
model = torch.nn.MultiheadAttention(embed_dim=64, num_heads=1)
model = model.eval()
with torch.no_grad():
    out1 = model(in2, in1, in1)[0]
    out2 = model(in2, in1, in1)[0]
    print(torch.equal(out1, out2))  # prints True, just a sanity check
    out3 = torch.cat([model(in2[:64], in1, in1)[0], model(in2[64:], in1, in1)[0]])
    print(torch.equal(out1, out3))  # prints False
    out4 = torch.cat([model(in2[:100], in1, in1)[0], model(in2[100:], in1, in1)[0]])
    print(torch.equal(out1, out4))  # prints False

np.random.seed(12)
in1 = torch.FloatTensor(np.random.rand(256**2, 1, 64))
in2 = torch.FloatTensor(np.random.rand(128, 1, 64))
model = torch.nn.MultiheadAttention(embed_dim=64, num_heads=1)
model = model.eval()
with torch.no_grad():
    out1 = model(in2, in1, in1)[0]
    print(torch.equal(out1[:64], model(in2[:64], in1, in1)[0]))  # prints False 
    print(torch.equal(out1[:128], model(in2[:128], in1, in1)[0]))  # prints True

a = F.linear(in2, model.in_proj_weight, model.in_proj_bias)[:64]
b = F.linear(in2[:64], model.in_proj_weight, model.in_proj_bias)
print(a.dist(b))

import torch
import numpy as np

def apply_chunked(model, q, k, v, chunk_size):
    parts = []
    for start_idx in torch.arange(0, q.shape[0], chunk_size):
        end_idx = start_idx + chunk_size
        chunk_q = q[start_idx:end_idx]
        parts.append(model(chunk_q, k, v)[0])
    return torch.cat(parts)
    

torch.manual_seed(12)
np.random.seed(12)
for embedding in [64, 96, 128, 144]:
    print(f'embedding: {embedding}')
    in1 = torch.FloatTensor(np.random.rand(256**2, 1, embedding))
    in2 = torch.FloatTensor(np.random.rand(256, 1, embedding))
    for num_heads in [1,4,8,16]:
        for chunk_size in [16, 32, 64, 96, 128, 140]:
            model = torch.nn.MultiheadAttention(embed_dim=embedding, num_heads=num_heads)
            model = model.eval()
            with torch.no_grad():
                out_true = model(in2, in1, in1)[0]
                print(f'\tnum_heads={num_heads}\tchunk_size={chunk_size}\t'
                      f'verdict_joined={torch.equal(out_true, apply_chunked(model, in2, in1, in1, chunk_size))}\t'
                      f'verdict_prefix={torch.equal(out_true[:chunk_size], model(in2[:chunk_size], in1, in1)[0])}')
        print()