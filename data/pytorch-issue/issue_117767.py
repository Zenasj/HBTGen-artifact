import torch

p = torch.rand(2, 3, device="cuda")
p.grad = torch.rand_like(p)
optim = torch.optim.AdamW([p])

def f():
    optim.step()

torch.compile(f, backend="eager")()