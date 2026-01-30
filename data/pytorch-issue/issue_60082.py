import torch.nn as nn

import torch

matrix = torch.nn.Parameter(torch.zeros(2, 1, device="cuda"))
batch = torch.zeros(1, 2, device="cuda")
optimizer = torch.optim.Adam([matrix])

for _ in range(50):
    loss = (batch @ matrix @ matrix.t()).sum()
    loss.backward()
    optimizer.step()