import torch.nn as nn

import torch

def f(x):
    y = x + 1
    z = torch.nn.Parameter(y)
    with torch.no_grad():
        z.mul_(2)
    return y + z

x = torch.ones(2, requires_grad=True)
out_ref = f(x)
out_test = torch.compile(f, backend='aot_eager')(x)
print(out_ref)
print(out_test)
print(torch.allclose(out_ref, out_test))