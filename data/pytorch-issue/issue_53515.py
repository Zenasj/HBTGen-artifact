import torch.nn as nn

# EXAMPLE 1
import torch
from torch import nn, jit
from torch.optim import SGD


inputs = torch.tensor([2.0], device="cuda")
model = nn.Linear(1, 1, bias=False).to("cuda")

optimizer = SGD(model.parameters(), lr=1e-1)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, x):
        param = next(self.parameters())

        param.requires_grad = True
        x = self.model(x).mean()
        param.requires_grad = False
        return x


c = MyModule()
forward = jit.trace(c, (inputs,))
result = forward(inputs)

result.mean().backward()

optimizer.step()
optimizer.zero_grad()

print("It does not work fine!")

# EXAMPLE 2
import torch
from torch import nn, jit
from torch.optim import SGD


inputs = torch.tensor([2.0], device="cuda")
model = nn.Linear(1, 1, bias=False).to("cuda")

optimizer = SGD(model.parameters(), lr=1e-1)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, x):
        param = next(self.parameters())

        param.requires_grad = False # True --> False
        x = self.model(x).mean()
        param.requires_grad = True # False --> True
        return x


c = MyModule()
forward = jit.trace(c, (inputs,))
result = forward(inputs)

result.mean().backward()

optimizer.step()
optimizer.zero_grad()

print("It does work fine!")