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

primal = torch.ones(1)
tangent = torch.ones(1)

with fwAD.dual_level():
    y = fwAD.make_dual(WrapperTensor(primal), tangent)
    print(y)

WrapperTensor(tensor([1.]))

import torch
from torch.utils._pytree import tree_map
import torch.autograd.forward_ad as fwAD

from utils import no_dispatch

class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, e):
        r = torch.Tensor._make_subclass(cls, e)
        r.elem = e
        return r

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        with no_dispatch():
            return f'WrapperTensor(self={super().__repr__()}, elem={self.elem})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        print(func)
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

primal = torch.ones(1)
tangent = torch.ones(1)

with fwAD.dual_level():
    y = fwAD.make_dual(WrapperTensor(primal), tangent)
    print(y)

OpOverloadPacket(op='aten.alias')
OpOverloadPacket(op='aten._has_same_storage_numel')
WrapperTensor(self=tensor([1.], tangent=tensor([1.])), elem=tensor([1.]))