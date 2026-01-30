import torch.nn as nn

import torch

class MinimalExample(torch.nn.Module):
    def __init__(self):
        super(MinimalExample, self).__init__()

    def forward(self, x):
        perm = torch.randperm(x.numel())
        print(perm)
        return x[perm]

model = MinimalExample()
x = torch.rand(3)
# works fine
print(model(x))

jit_model = torch.jit.script(model)
# error!
print(jit_model(x))