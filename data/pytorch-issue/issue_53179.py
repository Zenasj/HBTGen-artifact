import torch

from torch import tensor
from torch.nn import Module, Parameter, ParameterList


# More realistic case
class MyModule(Module):
    def __init__(self):
        super().__init__()
        params = [Parameter(tensor(1.0)) for _ in range(2)]
        self.list = ParameterList(params)



m = MyModule()
m.train()


# Minimal case
list = ParameterList(Parameter(tensor(1.0)) for _ in range(2))
list.train()