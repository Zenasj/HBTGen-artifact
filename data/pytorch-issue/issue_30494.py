import torch

@torch.jit.script
def f():
    l = []
    for n in [2, 1]:
        l.append(torch.zeros(n))

    return l[0]

f()
f()

def f():
    l = []
    for n in [2, 1]:
        l.append(torch.zeros(n))

    return l[0]