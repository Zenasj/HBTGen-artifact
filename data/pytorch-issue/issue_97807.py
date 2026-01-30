import torch.nn as nn

import torch
x = torch.rand([1,2,1], device='cuda')
def forward():
    a = torch.nn.functional.pad(x, (0, 1))
    b = torch.nn.functional.pad(a, (0, 0, 0, 1), 'reflect')
    b[0, 0, 0] = 0.1
    return b

with torch.no_grad():
    print(forward())
    fn_compiled = torch.compile(forward)
    print(fn_compiled())