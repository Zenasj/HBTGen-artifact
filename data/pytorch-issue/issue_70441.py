import torch.nn as nn

3
from torch import nn

a = [nn.Module()]
b = [nn.Module()]
c = a + b
print(c)

a = nn.ModuleList([nn.Module()])
b = nn.ModuleList([nn.Module()])
c = a + b # TypeError: unsupported operand type(s) for +: 'ModuleList' and 'ModuleList'
print(c)

class ModuleList(Module):
    def __add__(self, other: 'ModuleList') -> 'ModuleList':
        r"""Concat two ModuleList instances.
        Args:
            other (ModuleList): modulelist to add
        """
        if not isinstance(other, ModuleList):
            raise TypeError("ModuleList concatenation should only be "
                            "used with another ModuleList instance, but "
                            " got " + type(other).__name__)
        offset = len(self)
        concat = ModuleList()
        for i in range(len(self)):
            concat.add_module(str(i), self[str(i)])
        for i in range(len(other)):
            concat.add_module(str(offset + i), other[str(i)])
        return concat