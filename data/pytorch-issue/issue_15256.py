import torch

def f(a):
    b, c = torch.unique(a, return_inverse=True)
    return b.sum()

def g(a):
    b = torch.unique(a)
    return b.sum()

a = torch.rand((5, 6, 7))

traced_f = torch.jit.trace(f, a)
traced_g = torch.jit.trace(g, a)
print('traced f:', traced_f.graph)
print('traced g:', traced_g.graph)

scripted_f = torch.jit.script(f)
print('scripted f:', scripted_f.graph)

scripted_g = torch.jit.script(g)
print('scripted g:', scripted_g.graph)