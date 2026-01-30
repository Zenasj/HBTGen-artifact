import torch.nn as nn
import random

import torch
import numpy as np
import gc

model = torch.nn.Sequential(
    torch.nn.Linear(3, 3),
    torch.nn.Linear(3, 3),
    torch.nn.Linear(3, 3),
    torch.nn.Linear(3, 3),
    torch.nn.Linear(3, 3),
    torch.nn.Linear(3, 3),
    )

optimizer = torch.optim.SGD(model.parameters(), lr=.01)

running_loss = 0.


def my_fn():
    # 228.9 MB
    f0, f1 = np.random.rand(3).astype(np.float32), np.random.rand(3).astype(np.float32)
    frames = torch.stack([
        torch.from_numpy(f0),
        torch.from_numpy(f1)
    ])

    # 230 MB
    # f0, f1 = torch.rand(3), torch.rand(3)
    # frames = torch.stack([
    #     f0,
    #     f1
    # ])

    # 230 MB
    # frames = torch.rand((2,3))

    optimizer.zero_grad()
    loss = model(frames).sum()
    loss.backward()
    optimizer.step()

    # Adding .item() takes 150 MB for all cases above
    return loss


for n in range(10000):
    loss = my_fn()
    running_loss += loss

# Change to input() if you use python 3
raw_input("Check memory now")