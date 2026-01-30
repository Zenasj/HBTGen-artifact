import torch.nn as nn

import torch

class MyTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __mul__(self, rhs):
        return self.tensor * rhs

class MyModule(torch.nn.Module):
    def forward(self, t: torch.Tensor):
        my_tensor = MyTensor(torch.ones_like(t))
        return my_tensor * t # Fails to trace this

my_module = MyModule()

torch.export.export(my_module, args=(torch.tensor([2, 3]),))

import torch

class MyTensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def my_mul(self, rhs):
        return self.tensor * rhs

class MyModule(torch.nn.Module):
    def forward(self, t: torch.Tensor):
        my_tensor = MyTensor(torch.ones_like(t))
        return my_tensor.my_mul(t)

my_module = MyModule()

torch.export.export(my_module, args=(torch.tensor([2, 3]),))