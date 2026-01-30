py
import torch

class NJT:
    def __repr__(self):
        return f"NJT(shape={self.shape})"

    def __init__(self, values, offsets):
        self._values = values
        self._offsets = offsets

    def sin(self):
        return torch.sin(self)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == torch.sin:
            self = args[0]
            return NJT(func(self._values), self._offsets)
        if func == torch.mul:
            x, y = args
            assert isinstance(x, NJT) and isinstance(y, NJT)
            return NJT(torch.mul(x._values, y._values), x._offsets)
        raise AssertionError("not implemented")

values1 = torch.randn(10, 3, 4, requires_grad=True)
values2 = torch.randn(10, 3, 4, requires_grad=True)
values3 = torch.randn(10, 3, 4, requires_grad=True)
values4 = torch.randn(10, 3, 4, requires_grad=True)
offsets = torch.tensor([0, 3, 10])
njt1 = NJT(values1, offsets)
njt2 = NJT(values2, offsets)
njt3 = NJT(values1, offsets)
njt4 = NJT(values2, offsets)

@torch.compile(backend="eager", fullgraph=True)
def f(x):
    return x.sin()

res = f(njt1)
res = f(njt2)