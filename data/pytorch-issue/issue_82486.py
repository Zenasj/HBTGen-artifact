import torch.nn as nn

import torch

net = torch.nn.Embedding(5, 1, padding_idx=0, sparse=True)

# Simulate a batch that only indexes the embedding at padding_idx
x = torch.tensor([[0, 0]]).int()
y = torch.tensor([[3.0, 4.0]])

adam = torch.optim.SparseAdam(net.parameters())

loss_fn = torch.nn.MSELoss()

loss = loss_fn(net.forward(x), y)

loss.backward()

adam.step()  # RuntimeError: values has incorrect size, expected [0, 1], got [0, 0]