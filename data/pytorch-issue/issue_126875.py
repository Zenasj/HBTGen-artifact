import torch.nn as nn
import torch.nn.functional as F

def __getattr__(self, name):
    try:
        super().__getattr__(name)
    except AttributeError:
        bla

import torch
from torch.nn import functional as F
from torch import nn

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cst = nn.Parameter(torch.zeros(()))
        self.linear = torch.nn.Linear(10, 10)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.__dict__["_modules"]["linear"], name)
            except KeyError:
                raise AttributeError

    def forward(self, input):
        return F.linear(input, self.linear.weight, self.linear.bias) + self.cst # Works 
        # return F.linear(input, self.weight, self.bias) + self.cst  # Fails

module = MyModule()
x = torch.randn(10)
module(x)
module_c = torch.compile(module, fullgraph=True)
module_c(x)