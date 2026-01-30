import torch.nn as nn

import torch

class FooBar(torch.nn.Module):
    __overloads__ = {'kek': ['top_kek']}

    def kek(self):
        return True

    @torch.jit.export
    def top_kek(self):
        return False

    @torch.jit.export
    def test(self):
        return self.kek()

    def forward(self):
        return torch.rand(3, 4)

f = FooBar()
torch.jit.script(f)

class FooBar(torch.nn.Module):
    @torch.jit._overload_method
    def kek(self):
        # type: () -> bool
        pass

    @torch.jit.export
    def kek(self):
        return True

    @torch.jit.export
    def test(self):
        return self.kek()

    def forward(self):
        return torch.rand(3, 4)

f = FooBar()
torch.jit.script(f)