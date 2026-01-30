import torch
import operator


torch.manual_seed(11)


def f(x):
    return operator.getitem(x, [slice(0, 3), slice(0, 4)])


class BasicTensorSubclass(torch.Tensor):
    pass


x = BasicTensorSubclass(torch.rand(5, 5))

cf = torch.compile(f, fullgraph=False)
print(cf(x))  # SUCCEEDS!

torch.compiler.reset()

cf = torch.compile(f, fullgraph=True)
print(cf(x))  # FAILS!