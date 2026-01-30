import torch.nn as nn

import torch

class M(torch.nn.Module):
    def __init__(self, linear, encode=False):
        super().__init__()  # graph breaks
        self.linear = linear
        self.encode = encode
        self.linear.requires_grad_(False)
        if self.encode:
            self.linear.half()
        self.linear.requires_grad_(True)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            if self.encode:
                x = x.half()
            y = self.linear(x)
            y = y.sum()
        return y

x = torch.rand(5, 5, dtype=torch.float32, device="cuda", requires_grad=True)

def train(x, encode=True):
    linear = torch.nn.Linear(5, 5, device="cuda")
    # linear.requires_grad_(False)
    m = M(linear, encode=encode)
    with torch.cuda.amp.autocast(enabled=False):
        y = m(x)
        y.backward()
    return m.linear.weight.grad

print(train(x, True))

compiled = torch.compile(train, dynamic=True)
print(compiled(x, True))