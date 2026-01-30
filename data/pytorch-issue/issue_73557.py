import torch
from torch.utils._pytree import tree_map
import torch.autograd.forward_ad as fwAD

class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, e):
        r = torch.Tensor._make_wrapper_subclass(cls, e.shape, dtype=e.dtype, requires_grad=False)
        r.elem = e 
        return r 

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __str__(self):
        return f'WrapperTensor({self.elem})'

    def __repr__(self):
        return str(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        def unwrap(e):
            if isinstance(e, WrapperTensor):
                return e.elem
            else:
                return e 

        def wrap(e):
            if isinstance(e, torch.Tensor):
                return WrapperTensor(e)
            else:
                return e 

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

primal = torch.tensor(1)

with fwAD.dual_level():
    x = fwAD.make_dual(primal, torch.tensor(2))
    y = fwAD.make_dual(WrapperTensor(x), torch.tensor(3))
    print(fwAD.unpack_dual(y))

UnpackedDualTensor(primal=WrapperTensor(1), tangent=tensor(3))

x = torch.tensor(1)
wrapped_x = WrapperTensor(x)
wrapped_y = wrapped_x.view_as(wrapped_x)
print(wrapped_y._base is wrapped_x)   # True
print(wrapped_y.elem._base is wrapped_x.elem)  # True (edited: previously this was False due to a typo in the code)