import random

import torch

device = "cpu"
loop = 10
scale = 123456000
rows = 10
left = 3
right = 3
columns = left + right

for i in range(loop):
    values = scale * torch.randn(
        (rows, columns), dtype=torch.float32, device=device
    )
    d = torch.allclose(values[:, :columns].mean(0)[:left], values[:, :left].mean(0))
    print(d, values[:, :columns].mean(0)[:left] - values[:, :left].mean(0))

import numpy as np

device = "cpu"
loop = 10
scale = 123456000
rows = 10
left = 3
right = 3
columns = left + right

for i in range(loop):
    values = scale * np.random.randn(
        rows, columns
    ).astype(np.float32)
    d = np.allclose(values[:, :columns].mean(0)[:left], values[:, :left].mean(0))
    print(d, values[:, :columns].mean(0)[:left] - values[:, :left].mean(0))