import torch.nn as nn

import torch

class MyModel(torch.nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        return torch.nn.Dropout(0.1)(self.linear(x))

x = torch.randn(5, 5)

m = MyModel()
print(m(x))

opt_m = torch.compile(backend="eager")(m)
print(opt_m(x))

Dropout

training

None

Dropout.__init__

p

inplace

Dropout

training

nn.Module

object_new

object_new