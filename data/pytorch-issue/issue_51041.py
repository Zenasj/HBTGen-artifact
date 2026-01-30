import torch
import torch.nn as nn

# This throws the same error, because `self.foo` is empty

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.foo: Dict[int, str] = {}

    def forward(self, x: Dict[int, str]):
        self.foo = x
        return 1

# This works fine

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.foo: Dict[int, str] = {1: "foo", 2: "bar", 3: "baz"}

    def forward(self, x: Dict[int, str]):
        self.foo = x
        return 1