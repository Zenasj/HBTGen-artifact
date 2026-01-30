import torch

class Base(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        for i in range(x.size(0)):
            x += x[i].sum()
        return x


class Derived(Base):
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        for i in range(x.size(0)):
            x -= x[i].sum()
        return x

d = Derived()
print(d(torch.rand(3, 4)))