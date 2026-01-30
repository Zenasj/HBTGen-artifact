import torch
import torch.nn as nn

n, d, m = 3, 5, 7
batch_size = 11

embedding = nn.Embedding(n, d, max_norm=True)
W = torch.randn((m, d), requires_grad=True)
optimizer = torch.optim.Adam(list(embedding.parameters()) + [W], lr=1e-3)

optimizer.zero_grad()
idx = torch.tensor([1, 2])

a = embedding.weight @ W.t()  # Line a 
b = embedding(idx) @ W.t()    # Line b

out = (a.unsqueeze(0) + b.unsqueeze(1))
loss = out.sigmoid().prod()
loss.backward()
optimizer.step()