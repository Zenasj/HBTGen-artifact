import torch.nn as nn

import torch

n = 65536
m = 50257


a = torch.randn(n, m, device='cuda', requires_grad=True)
b = torch.randint(0, m, (n,), device='cuda')
c = torch.nn.CrossEntropyLoss()

loss = c(a, b)
print(loss)
loss.backward()
print("Passed")