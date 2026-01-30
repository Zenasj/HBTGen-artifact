import torch.nn as nn

import torch
import torch._dynamo

class Good1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view((4, 4, -1))
        x = x.transpose(-1, -2)
        x /= 2.0
        return (x,)

class Good2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=4, out_features=16, bias=False).cuda()

    def forward(self, x):
        x = self.lin(x)
        x = x.view((4, 4, -1))
        x = x.transpose(-1, -2)
        return (x,)

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=4, out_features=16, bias=False).cuda()

    def forward(self, x):
        x = self.lin(x)
        x = x.view((4, 4, -1))
        x = x.transpose(-1, -2)
        x /= 2.0
        return (x,)

if __name__ == '__main__':
    inp1 = torch.randn(4, 4, 4, device="cuda")
    torch.compile(Good1())(inp1)

    inp2 = torch.randn(4, 4, device="cuda")
    torch.compile(Good2())(inp2)

    inp3 = torch.rand(4, 4, device="cuda")
    mod = Repro()
    mod(inp3)
    opt_mod = torch.compile(mod)
    opt_mod(inp3)