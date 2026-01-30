import torch
import torch.nn as nn

def forward(self, x):  # x: [s0, s1]
    return x.reshape([-1, x.shape[0] - 1])  # Eq(Mod(s0 * s1, s0 - 1), 0)

class Foo(torch.nn.Module):
    def forward(self, x, y):
        # check that negation of first guard also shows up as runtime assertion
        if x.shape[0] == y.shape[0]:  # False
            return x + y
        elif x.shape[0] == y.shape[0] ** 3:  # False
            return x + 2, y + 3
        elif x.shape[0] ** 2 == y.shape[0] * 3:  # True
            return x * 2.0, y * 3.0