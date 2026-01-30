import torch.nn as nn

import torch

def forward():
    a = torch.tensor([2.], device='cuda:0')
    b = torch.nn.functional.pad(a, (0, 0)) # cloned ?
    if a.mean() >= -1e5:
        a[0] = 1
    return b

with torch.no_grad():
    print(forward()) # 2.
    fn_compiled = torch.compile(forward)
    print(fn_compiled()) # 1.