import torch

lib = torch.library.Library("bar", "FRAGMENT")

lib.define("foo(Tensor x) -> Tensor")

def foo_impl(a):
    return a.clone()

lib.impl("foo", foo_impl, "CPU")
lib.impl("foo", foo_impl, "CUDA")
lib.impl("foo", foo_impl, "Meta")

from torch._higher_order_ops.effects import (
    _EffectType,
    _register_effectful_op,
)

_register_effectful_op(
    torch.ops.bar.foo.default, _EffectType.ORDERED
)

class DoubleSizeMaybeAddGeThreeTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, inner):
        # Double the outer-most dimension
        outer_shape = (inner.shape[0] * 2,) + inner.shape[1:]
        return torch.Tensor._make_wrapper_subclass(
            # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
            # Calling the overload that has kwargs causes us to go down the first overload path,
            # which will **always** specialize sizes.
            # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
            cls,
            outer_shape,
            inner.stride(),
            None,
            None,
            inner.dtype,
            inner.layout,
            inner.device,
            False,
            inner.requires_grad,
        )

    def __init__(self, inner):
        self.inner_elem = inner

    def __tensor_flatten__(self):
        return ["inner_elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, _, outer_size, outer_stride):
        return DoubleSizeMaybeAddGeThreeTensor(inner_tensors["inner_elem"])

    def __repr__(self):
        return f"DoubleSizeMayberAddGeThreeTensor({repr(self.inner_elem)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args_inner = torch.utils._pytree.tree_map_only(
            DoubleSizeMaybeAddGeThreeTensor, lambda x: x.inner_elem, args
        )
        out_inner = func(*args_inner, **kwargs)

        # Add guards on the  inner tensor's sizes
        if args_inner[0].shape[0] > 3:
            out_inner += 2

        return DoubleSizeMaybeAddGeThreeTensor(out_inner)


def fn(x):
    return torch.ops.bar.foo(x)


compiled_fn = torch.compile(fn, backend="aot_eager")
wrapped_x = DoubleSizeMaybeAddGeThreeTensor(torch.tensor([1., 2., 3.]))
out = compiled_fn(wrapped_x)