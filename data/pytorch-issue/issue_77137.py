import torch
import torch.nn as nn
import torch.nn.utils.stateless as stateless

class Foo(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(5))

    def forward(self, x):
        return self.weight + x

mod = Foo().cuda()
mod = nn.DataParallel(mod, [0, 1])
print(stateless.functional_call(mod, {'module.weight': torch.zeros(5, device='cuda')}, (torch.ones(2, 5, device='cuda'),)))