import torch
import torch.nn as nn

class Module(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear  

    def forward(self, x):
        self.linear.requires_grad_(False)   # This performs set __contains__ op by memoizing its search of params.
        return self.linear(x)


def create(x):
    linear = torch.nn.Linear(3, 3)  # This becomes an `UnspecializedNNModuleVariable`
    mod = Module(linear)
    return mod(x)

x = torch.zeros(3)
print(torch.compile(create, backend="eager")(x))  # num graph breaks = 2