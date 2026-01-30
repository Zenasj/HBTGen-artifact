import torch

def f(x):
    return torch.ones(3, 2).matmul(x.unsqueeze(-1)).squeeze(-1).sum()

jf = torch.jit.trace(f, torch.ones(2,))
x = torch.ones(2, requires_grad=True)
jf(x).backward()