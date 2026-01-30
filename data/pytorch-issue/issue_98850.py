import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        z = torch.cat([x, y])
        z = torch.relu(z)
        return z

x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)

func = Model()

jit_func = torch.compile(func)

print(func(x, y))
print(jit_func(x, y))