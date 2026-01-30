import torch.nn as nn

import torch

class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        r0 = x.add(y)
        return r0

device = 'new_device'
lhs = torch.randn(1, 3, 5, 5).to(device)
rhs = torch.randn(1, 3, 5, 5).to(device)

mod = SimpleModule().to(device)
optimized_mod = torch.compile(mod)

res = optimized_mod(lhs, rhs)
res.to('cpu')
print(f'res_shape={res.shape}\noptimized_res={res}')