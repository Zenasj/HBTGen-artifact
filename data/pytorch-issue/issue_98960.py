import torch
import torch.nn as nn


@torch.compile(backend='aot_eager')
def f(x):
    return torch.masked.norm(x, 0, mask=None)

a = torch.tensor(0.8604, requires_grad=True)
out = f(a)
out.sum().backward()

import torch
import torch.nn as nn


def f(x):
    return torch.masked.norm(x, 0, mask=None)

a = torch.tensor(0.8604, requires_grad=True)
out = f(a)
out.sum().backward()
print(a.grad)  # prints None!