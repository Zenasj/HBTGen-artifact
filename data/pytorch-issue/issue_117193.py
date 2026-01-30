py
import torch
import torch.utils._pytree as pytree

class DoubleTensor(object):
    def __init__(self, value):
        self.value = value

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        new_args, new_kwargs = pytree.tree_map_only(DoubleTensor, lambda x: x.value, (args, kwargs))
        output = func(*new_args, **new_kwargs)
        return DoubleTensor(output * 2)

@torch.compile(backend="aot_eager", fullgraph=True)
def f(x):
    y = DoubleTensor(x)
    z = torch.mul(y, 1)
    return z

x = torch.tensor(1.)
out = f(x)
print(out.value)