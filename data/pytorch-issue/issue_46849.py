import torch.nn as nn

import torch
p = torch.ones(100, requires_grad=True, dtype=torch.float32)

print(torch.__version__, '\n')

for J in [
    p,
    p * 1e20,
    (p * torch.tensor(float('inf')))
]:
    J = J.sum()

    J.backward()
    print('max(abs(grad))\t', torch.max(torch.abs(p.grad)))
    print('grad_norm\t', torch.nn.utils.clip_grad_norm_(p, 1))
    print('max(abs(grad))\t', torch.max(torch.abs(p.grad)))
    print()
    p.grad = None