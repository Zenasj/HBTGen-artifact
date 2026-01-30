import torch.nn as nn

import torch
from torch import nn
device = torch.device('mps')
torch.set_default_device(device)


input_batch = torch.tensor([[1, 3, 4, 4, 8],
                            [2, 5, 7, 7, 9]], device=device, dtype=torch.int64)

# forward and backward for embedding_dim == 36
emb_36_a = nn.Embedding(embedding_dim=36, num_embeddings=100, device=device)
res_36_a = emb_36_a(input_batch)
loss_36_a = torch.sum(res_36_a) - 1
loss_36_a.backward()

# forward and backward again, but with a new nn.Embedding() but same embedding_dim == 36,
# (and presumably new weights)
emb_36_b = nn.Embedding(embedding_dim=36, num_embeddings=100, device=device)
res_36_b = emb_36_b(input_batch)
loss_36_b = torch.sum(res_36_b) - 1
loss_36_b.backward()

# forward and backward, but with a new nn.Embedding() and increasing embedding_dim == 48
emb_48 = nn.Embedding(embedding_dim=48, num_embeddings=100, device=device)
res_48 = emb_48(input_batch)
loss_48 = torch.sum(res_48) - 1
loss_48.backward()  # <--- Error