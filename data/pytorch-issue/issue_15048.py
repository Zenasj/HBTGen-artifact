import torch
import torch.nn as nn


class Block(torch.jit.ScriptModule):
    def __init__(self, dim):
        super().__init__()

        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    @torch.jit.script_method
    def forward(self, x):
        # x = x + self.linear(x)
        # x = x / 2.
        # return self.relu(x)
        return self.relu((self.linear(x) + x) / 2.)


m = Block(32)
x = torch.randn(1, 32)
print('cpu:', m(x))

m.cuda()
x = x.cuda()
print('gpu:', m(x))