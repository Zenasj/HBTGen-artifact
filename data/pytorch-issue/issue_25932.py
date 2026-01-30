import torch

py
x = torch.tensor(0., requires_grad=True)
with torch.no_grad():
    y = x.expand(1)
assert not y.requires_grad  # fails