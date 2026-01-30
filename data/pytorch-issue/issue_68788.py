import torch.nn as nn

import torch

class Thing(torch.nn.Module):
    torch.jit.export
    def en(self, x: torch.Tensor):
        return torch.add(x, 2.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        a = torch.mm(x, y)
        b = torch.nn.functional.gelu(a)
        c = self.en(b)
        return c.std_mean()

if __name__ == '__main__':
    unsc = Thing()
    thing = torch.jit.script(unsc)
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    std, mean = thing.forward(x, y)
    print(std, mean)
    print(str(thing.forward.graph))