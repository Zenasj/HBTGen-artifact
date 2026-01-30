import torch.nn as nn

import torch

class Test(torch.nn.Module):
    def __init__(self, add):
        super().__init__()
        self.add = add

    def forward(self, x):
        x += self.add
        return x

m = Test(1)
m.eval()
x = torch.randn(1, 1, 10)
m = torch.jit.trace(m, x)
m.save('Test.jit')

m = torch.jit.load('Test.jit')
m.eval()
x = torch.randn(1, 1, 10)
print(x)
with torch.no_grad():
    y = m(x)
    print(x)  # <-- x is changed