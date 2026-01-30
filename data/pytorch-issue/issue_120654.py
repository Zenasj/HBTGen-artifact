import torch
import torch.utils._pytree as pytree

class SubTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, t):
        return torch.Tensor._make_wrapper_subclass(
            cls, t.shape, t.stride(), t.storage_offset(), torch.contiguous_format, t.dtype,
            torch.strided, t.device, False, t.requires_grad, "sizes", False, False, None
        )

    def __init__(self, t):
        super().__init__()
        self._t = t

    def __tensor_flatten__(self):
        return ["_t"], {}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        t = inner_tensors["_t"]
        return SubTensor(t)

    def __repr__(self):
        return f"SubTensor({self._t})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        new_args = pytree.tree_map_only(SubTensor, lambda s: s._t, args)
        output = func(*new_args, **kwargs)
        output = pytree.tree_map_only(torch.Tensor, lambda t: SubTensor(t), output)
        return output


@torch.compile(dynamic=True)
def f(x):
    return x.unflatten(-1, [2, 5])

s = SubTensor(torch.randn(3, 10))
output = f(s)
print(output)

import torch
import torch.utils._pytree as pytree

class SubTensorTF:
    def __init__(self, t):
        super().__init__()
        self._t = t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        new_args = pytree.tree_map_only(SubTensorTF, lambda s: s._t, args)
        output = func(*new_args, **kwargs)
        output = pytree.tree_map_only(torch.Tensor, lambda t: SubTensorTF(t), output)
        return output

class SubTensorTD(torch.Tensor):
    @staticmethod
    def __new__(cls, t):
        return torch.Tensor._make_wrapper_subclass(
            cls, t.shape, t.stride(), t.storage_offset(), torch.contiguous_format, t.dtype,
            torch.strided, t.device, False, t.requires_grad, "sizes", False, False, None
        )

    def __init__(self, t):
        super().__init__()
        self._t = t

    def __tensor_flatten__(self):
        return ["_t"], {}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        t = inner_tensors["_t"]
        return SubTensorTD(t)

    def __repr__(self):
        return f"SubTensor({self._t})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        new_args = pytree.tree_map_only(SubTensorTD, lambda s: s._t, args)
        output = func(*new_args, **kwargs)
        output = pytree.tree_map_only(torch.Tensor, lambda t: SubTensorTD(t), output)
        return output

@torch.compile(dynamic=True)
def f(x):
    return torch.unflatten(x, -1, [2, 5])

s = SubTensorTF(SubTensorTD(torch.randn(3, 10)))
output = f(s)
print(output)