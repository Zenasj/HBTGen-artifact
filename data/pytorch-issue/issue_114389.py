import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

class SubclassTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a):
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, a):
        self.a = a

    def __repr__(self):
        a_repr = repr(self.a)
        return f"SubclassTensor({a_repr})"

    def __tensor_flatten__(self):
        return ["a"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        assert meta is None
        a = inner_tensors["a"]
        return SubclassTensor(a)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map(lambda x: x.a if isinstance(x, SubclassTensor) else x, args)
        kwargs_a = pytree.tree_map(lambda x: x.a if isinstance(x, SubclassTensor) else x, kwargs)
        out_a = func(*args_a, **kwargs_a)
        out = pytree.tree_map(lambda x: SubclassTensor(x) if isinstance(x, torch.Tensor) else x, out_a)
        return return_and_correct_aliasing(func, args, kwargs, out)



@torch.compile
def f(x):
    out = SubclassTensor(x)
    return out * out

x =torch.ones(2)
out = f(x)

class SubclassTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a):
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    @torch._dynamo.allow_in_graph
    def __init__(self, a):
        self.a = a

@torch._dynamo.allow_in_graph
def constr(x):
    return SubclassTensor(x)

import torch
import torch.utils._pytree as pytree

class MySubclass(torch.Tensor):
    @staticmethod
    def __new__(cls, a, val):
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, a.shape, **kwargs)
        out.a = a
        out.val = val
        return out

    def custom_add(self, y):
        return torch.add(self, y).add(self.val)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(MySubclass, lambda x: x.a, args)
        out_a = func(*args_a, **kwargs)
        return MySubclass(out_a, val=args[0].val)

@torch.compile(backend='eager', fullgraph=True)
def f(x, y):
    return x.custom_add(y)

x = MySubclass(torch.ones(3), val=5)
y = torch.ones(3)
out = f(x, y)
print(out.a)