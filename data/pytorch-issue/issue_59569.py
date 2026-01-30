import torch
import torch.nn as nn

class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a=None, b=None):
        res = a
        if b is not None:
            res = res + b
        return res

concrete_args = {'b': torch.tensor(5)}
traced = fx.symbolic_trace(Foo(), concrete_args=concrete_args)