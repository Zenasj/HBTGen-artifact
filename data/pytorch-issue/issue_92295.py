import functorch
import torch
import torch.nn as nn

import optree

class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        nn.init.ones_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = self.fc(x).mean()
        grad = torch.autograd.grad(out, self.parameters(), create_graph=True)
        return grad

model = Model(4)
x = torch.ones(4)