import torch

@torch.jit.script
def foo(x, y):
    if x > y:
        r = x
    else:
        r = x + y
    return r
foot.save("foo.pt")