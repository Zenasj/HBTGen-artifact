import torch

def forward(self, x1, x2, x3, y):
    z1 = x1.item()
    z2 = x2.item()
    z3 = x3.item()
    torch._check((z2 + z3) == z1)
    # torch._check(z1 == (z2 + z3)) didn't work, now does
    if z2 + z3 == z1:
        return y * 2
    else:
        return y + 3