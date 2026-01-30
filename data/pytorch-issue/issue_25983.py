import torch

x = torch.randn(5, 3, 2)
first = x[:,:1,:1]
print(first.mean())
x[...,:1] = x[...,:1] - first
print(first.mean())

tensor(-0.0770)
tensor(0.)

x = torch.randn(5, 3)
first = x[:,:1]
print(first.mean())
x = x - first
print(first.mean())

tensor(0.5736)
tensor(0.5736)