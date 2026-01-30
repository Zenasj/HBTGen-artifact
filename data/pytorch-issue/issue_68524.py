import torch.nn as nn

import torch
import pdb

class phi(torch.nn.Module):
    def __init__(self):
        super(phi,self).__init__()
    def forward(self,x):
        return (torch.sin(x.real)*torch.cosh(x.imag)).type(torch.complex64) +1.0j*(torch.cos(x.real)*torch.sinh(torch.maximum(x.imag,-5*torch.ones_like(x.imag)))).type(torch.complex64)


class testnet(torch.nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.l1 = torch.nn.Linear(1, 2,dtype=torch.complex64,bias=False)
        self.act = phi()
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        return x

model = testnet()

for param in model.parameters():
    print(param)

tx = torch.randn(1, dtype=torch.complex64, requires_grad=False)
complex_opt = torch.optim.Adam(model.parameters())

xz = torch.randn(2,1,dtype=torch.complex64)
yz = torch.randn(2,2,1,dtype=torch.complex64)
print(xz)
print(yz)

complex_opt.zero_grad()
for param in model.parameters():
    print(param.grad)
torch.square((yz-model(xz).view((2,2,1))).abs()).mean().backward()

for param in model.parameters():
    print(param.grad)

complex_opt.step()