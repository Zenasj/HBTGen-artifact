import torch

t = torch.full((2, 2), 1., requires_grad=True)

def fn():
    return torch.full((2, 2,), 1., requires_grad=True)

scripted_fn = torch.jit.script(fn)
scripted_fn()