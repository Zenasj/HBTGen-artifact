import torch.nn as nn

import torch

class MyNonLazyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = 100
    def forward(self, a, b=1):
        return a + b + self.c

m = MyNonLazyModule()

m(1, b=2)  # gives 103
m(a=1)  # gives 102
m(a=1, b=2)  # gives 103

import torch

class MyLazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = 100
    def initialize_parameters(self, a, b=1):
        if b == 1:
            self.c = 100
        else:
            self.c = 200
    def forward(self, a, b=1):
        return a + b + self.c

m = MyLazyModule()

m(1, b=2)  # gives 103, but it should be 203 if we strictly follow the `initialize_parameters` logic.
m(a=1)  # fails with "TypeError: initialize_parameters() missing 1 required positional argument: 'a'".
m(a=1, b=2)  # fails with "TypeError: initialize_parameters() missing 1 required positional argument: 'a'".