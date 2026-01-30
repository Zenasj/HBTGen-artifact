import torch

py
a = torch.tensor(1., requires_grad=True)
b = a.clone()
with th.no_grad():
    c = b.expand((1,))

c.requires_grad # True