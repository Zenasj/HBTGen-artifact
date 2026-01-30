import torch

@torch.jit.script
def scriptfn(t):
    t.requires_grad_(True)
    t_sq = t ** 2
    grads = torch.autograd.grad([t_sq], [t])
    t.requires_grad_(False)
    return grads


u = torch.tensor(2.)
print(scriptfn(u))