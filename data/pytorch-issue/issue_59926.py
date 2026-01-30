class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)

model = Model()
model.linear.register_buffer('mask', torch.ones(model.linear.weight.shape))

class WeightMaskParametrization(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        
    def forward(self, w):
        return self.layer.mask * w

weight_mask = WeightMaskParametrization(model.linear)
parametrize.register_parametrization(model.linear, 'weight', weight_mask);  # Notise the semicolon here

print(model)  # <- this will throw a recursion error

class WeightMaskParametrization(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        
    def forward(self, w):
        return self.mask * w

weight_mask = WeightMaskParametrization(model.linear.mask)
parametrize.register_parametrization(model.linear, 'weight', weight_mask)

weight_mask = WeightMaskParametrization(model.linear.mask)
parametrize.register_parametrization(model.linear, 'weight', weight_mask)
model.linear.mask = torch.zeros(model.linear.weight.shape)

model.linear.weight  # <- This will use the old mask tensor

import torch

a = torch.nn.Linear(2, 2)
b = torch.nn.Linear(2, 2)

a.b = b
b.a =a 

print(a)  # RecursionError

model.linear.mask.set_(torch.zeros(model.linear.weight.shape))

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

class WeightMaskParametrization(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, w):
        return self.mask * w

model = nn.Linear(3, 4)
model.register_buffer('mask', torch.ones_like(model.weight))
parametrize.register_parametrization(model, 'weight', WeightMaskParametrization(model.mask))

print(model.weight)   # print original weight
model.mask.set_(torch.zeros_like(model.weight))
print(model.weight)   # print zeros