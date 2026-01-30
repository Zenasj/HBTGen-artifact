import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1,bias=False)

    def forward(self, x):
        return self.linear(self.linear(x))
with torch.no_grad():
    m=M()
    m.linear.weight[:]=10.
    x=torch.tensor([2.])
    exported= torch.export.export(m, (torch.ones(1),))
    unflattened = torch.export.unflatten(exported)
    print(unflattened.graph)
    print(unflattened.linear.graph)
    assert m(x)==200.0
    assert unflattened(x)==200.0

def forward(self, x,y):
    z=[]
    for v in [x,y]:
        z.append(self.linear(v))
    return torch.cat(z,dim=0)

import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1,bias=False)

    def forward(self, x,y):
        z=[]
        for v in [x,y,x,y,x,y,x,y]:
            z.append(self.linear(v))
        return torch.cat(z,dim=0)

with torch.no_grad():
    m=M()
    m.linear.weight[:]=10.
    x=torch.tensor([2.])
    y = torch.tensor([3.])
    exported= torch.export.export(m, (x, y))
    unflattened = torch.export.unflatten(exported)
    print(unflattened.graph)
    print(unflattened.linear.graph)
    print(m(x, y))
    print(exported.module()(x,y))
    print(unflattened(x, y))