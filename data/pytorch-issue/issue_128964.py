# torch.rand(10, 3, 4, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn

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

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.offsets = torch.tensor([0, 3, 10])

    def forward(self, x):
        njt = NJT(x, self.offsets)
        res = njt.sin()
        return res._values  # Return the underlying tensor after operation

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 3, 4, dtype=torch.float32, requires_grad=True)

