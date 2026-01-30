import torch.nn as nn

import random
import torch
from torch import nn
import torch._dynamo.testing

class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a, b),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            *[nn.Linear(10, 10000), nn.ReLU()]
            + [nn.Linear(10000, 10000), nn.ReLU()]
            + [MyModule(10000, 10000)]
            + [MyModule(10000, 1000)]
        )

        self.layers = [
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
            MyModule(1000, 1000),
        ]
       
        self.tail = nn.Sequential(
            *[nn.Linear(1000, 5)]
        )

        self.layerdrop = 0.5

    def forward(self, x):
        hidden_state = self.pre(x)
        for layer in self.layers:

            dropout_probability = random.uniform(0, 1)

            if dropout_probability < self.layerdrop:
                continue

            hidden_state = layer(hidden_state) 
        
        hidden_state = self.tail(hidden_state)
        return hidden_state


m = ToyModel()
inputs = (torch.randn(20, 10),)
cnt = torch._dynamo.testing.CompileCounter()
opt_m = torch._dynamo.optimize(cnt)(m)
out = opt_m(*inputs)
assert cnt.frame_count == 1, "captured a full graph on the first try"

out = opt_m(*inputs)
assert cnt.frame_count == 2, "Expected a recompile on each iteration, since there are 2^n combinations of layerdrop"

random.uniform

import torch
import random

@torch.compile(backend="eager")
def fn(x):
    return (x + 1) * random.uniform(0, 1)

x = torch.ones(2)
for _ in range(3):
    print(fn(x))

tensor([0.7291, 0.7291])
tensor([0.7291, 0.7291])
tensor([0.7291, 0.7291])

tensor([1.7749, 1.7749])
tensor([1.5697, 1.5697])
tensor([1.1214, 1.1214])