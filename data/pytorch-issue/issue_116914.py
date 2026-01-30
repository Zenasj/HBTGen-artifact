import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        aa = torch.tensor([[0], [1], [2]])
        aa.expand_as(x)
        return self.relu(aa)

x = torch.ones(3, 2)
exported_program = export(Model(), args=(x,))
unflattened_module = unflatten(exported_program)

unflattened_module(x)