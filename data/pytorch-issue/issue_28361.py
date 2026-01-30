import torch

class MyTensor(torch.Tensor):
    _additional_attribute = "Kartoffel"

a = MyTensor([0, 1, 2, 3])
# b should be a MyTensor object, with all class attributes passed through.
b = torch_function(a)

class Tensor:
    def __torch_function__(self, f, t, a, kw):
        if not all(issubclass(ti, TensorBase) for ti in t):
            return NotImplemented
        result = f._wrapped(*a, **kw)
        return type(self)(result)