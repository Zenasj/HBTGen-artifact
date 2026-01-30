import torch.nn as nn

python
import torch
import torch.nn.utils._stateless

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.l1(x)

mod = MyModule()
print('weight before', mod.l1.weight)
x = torch.rand((1, 1))
parameters = {"l1.weight": torch.nn.Parameter(torch.tensor([[1.0]])),
              "l1.bias": torch.nn.Parameter(torch.tensor([0.0]))}
res = torch.nn.utils._stateless.functional_call(mod, parameters, x)
print('Functional call input ', x, ' and result ', res)
print('weight after', mod.l1.weight)

torch.nn.utils.parametrizations.spectral_norm(mod.l1)
x = torch.rand((1, 1))
parameters = {"l1.parametrizations.weight.original": torch.nn.Parameter(torch.tensor([[1.0]])),
              "l1.bias": torch.nn.Parameter(torch.tensor([0.0]))}
res = torch.nn.utils._stateless.functional_call(mod, parameters, x)