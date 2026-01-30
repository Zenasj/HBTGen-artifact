import torch.nn as nn

3
import torch

class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.))

a = Module()
b = Module()

inputs = tuple(a.parameters())
print(type(inputs)) # <class 'tuple'>
(a.param + b.param).backward(inputs=inputs)
print('ok')

inputs = a.parameters()
print(type(inputs)) # <class 'generator'>
(a.param + b.param).backward(inputs=inputs) # TypeError: object of type 'generator' has no len()
print('ok')