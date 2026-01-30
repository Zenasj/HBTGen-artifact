import torch.nn as nn

import torch
import functools

def my_hook(grad, *, k=0):
    return grad + k

hook = functools.partial(my_hook, k=3)

class MyMod(torch.nn.Module):
    def forward(self, x):
        x.register_hook(hook)
        y = x.mul(2)
        z = y.mul(3)
        return (z,)

mod = MyMod()
x = torch.ones(4, requires_grad=True)

with torch.device("cuda"):
    out = torch.compile(mod, fullgraph=True)(x)
    print(f"{out}")

# THIS WORKS
mod = MyMod()
x = torch.ones(4, requires_grad=True, device="cuda")

out = torch.compile(mod, fullgraph=True)(x)
print(f"{out}")