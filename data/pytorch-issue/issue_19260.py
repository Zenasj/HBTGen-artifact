import torch.nn as nn

model = nn.Sequential(...)
model(a, b) # TypeError!
checkpoint_sequential(model, 1, a, b) # OK.

import torch
import torch.utils.checkpoint

class Two(torch.nn.Module):
    def forward(self, a, b):
        return a, b
model = torch.nn.Sequential(Two())

a = torch.rand(1)
b = torch.rand(1)

torch.utils.checkpoint.checkpoint_sequential(model, 1, a, b)  # OK
model(a, b)  # TypeError!
# TypeError: forward() takes 2 positional arguments but 3 were given

class NewSequential(torch.nn.Module):

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def forward(self, *args):
        for module in self.children():
            if not isinstance(args, tuple):
                args = (args,)
            args = module(*args)
        return args

model1 = NewSequential(NewSequential(a, b), NewSequential(c, d))
model2 = NewSequential(a, b, c, d)

assert model1(x) == model2(x)