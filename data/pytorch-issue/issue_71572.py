import torch.nn as nn

import torch
x = torch.rand(10).to('cuda')
y = torch.tensor([11]).to('cuda')
print(x[y])

import torch
try:
    x = torch.nn.Embedding(100, 10)

    y = torch.randint(low=0, high=101, size=(30, 77))
    z = x(y)
except IndexError as e:
    print(e)