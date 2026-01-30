import torch

base = torch.rand(10, requires_grad=True)
with torch.no_grad():
    view = base[1]
view.copy_(var)
torch.autograd.grad(base.sum(), var)  # <- what should it return?