import torch.nn as nn

import torch

model = torch.nn.Sequential(torch.nn.Embedding(1, 1, sparse=True))
optimizer = torch.optim.SparseAdam(model.parameters())

print(list(model.parameters()))
# [Parameter containing:
# tensor([[0.8340]], requires_grad=True)]