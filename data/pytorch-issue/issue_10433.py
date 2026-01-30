import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        return self.linear(x)

model = Model()
x = torch.zeros(1)
trace = torch.jit.trace(x)(model)
trace.save('foobar.pt')

import torch
model = torch.jit.load('foobar.pt')
x = torch.zeros(1)
z = model(x)
print(z)

import torch
model = torch.jit.load('foobar.pt')
x = torch.zeros(1)
with torch.no_grad():
    z = model(x)
print(z)