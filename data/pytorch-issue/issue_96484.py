import torch.nn as nn

import torch
a = torch.rand([2])
b = torch.rand([2])
def forward(a, b):
    a = torch.nn.functional.pad(a, (0, -1))
    c = a + b
    return c.min(0).values
fn_compiled = torch.compile(forward)
print(fn_compiled(a, b))