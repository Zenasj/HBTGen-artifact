import torch

@torch.compile()
def foo(c):
    # c is nested
    d = torch.tanh(c)
    e = torch.nested.to_padded_tensor(d, padding=0)
    # this is dense code
    g = torch.tanh(e)
    h = torch.sin(e)
    return h