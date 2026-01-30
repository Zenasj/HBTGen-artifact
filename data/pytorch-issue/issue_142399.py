import torch
from typing import NamedTuple

@torch.compile(backend="eager")
def f(my_tuple):
    return my_tuple.a + 1

class MyTuple(NamedTuple):
    a: torch.Tensor
    b: torch.Tensor

    def __getitem__(self, index):
        return MyTuple(a[index], b[index])

a = torch.randn(2)
b = torch.randn(2)

my_tuple = MyTuple(a, b)
out = f(my_tuple)

my_tuple2 = MyTuple(a, b)
out = f(my_tuple2)