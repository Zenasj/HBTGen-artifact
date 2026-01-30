import torch.nn as nn

import torch

def my_custom_backend(gm, example_inputs):
    print(gm.training)
    return gm.forward

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if self.training:
            return x + 2
        else:
            return x + 3

mod = MockModule()

optimized_mod = torch.compile(mod, backend=my_custom_backend)

mod.eval()
optimized_mod.eval()
print(optimized_mod(torch.zeros(10)))
print(optimized_mod(torch.zeros(10)))

mod.train()
optimized_mod.train()
print(optimized_mod(torch.zeros(10)))
print(optimized_mod(torch.zeros(10)))

mod.eval()
optimized_mod.eval()
print(optimized_mod(torch.zeros(10)))
print(optimized_mod(torch.zeros(10)))

True
tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
True
tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])