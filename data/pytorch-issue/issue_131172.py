import torch

with torch.library._scoped_library("_mylib", "FRAGMENT") as lib:
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

            _register_effectful_op(torch.ops._mylib.foo.default, _EffectType.ORDERED)

            class DoubleTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, inner):
                    outer_shape = inner.shape
                    if inner.ndim > 0:
                        outer_shape = (inner.shape[0] * 2,) + inner.shape[1:]
                    return torch.Tensor._make_wrapper_subclass(
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

                def __repr__(self):
                    inner_repr = repr(self.inner_elem)
                    return f"DoubleTensor({inner_repr})"

                def __tensor_flatten__(self):
                    return ["inner_elem"], None

                @staticmethod
                def __tensor_unflatten__(inner_tensors, _, outer_size, outer_stride):
                    return DoubleTensor(inner_tensors["inner_elem"])

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    if kwargs is None:
                        kwargs = {}

                    args_inner = torch.utils._pytree.tree_map_only(
                        DoubleTensor, lambda x: x.inner_elem, args
                    )
                    kwargs_inner = torch.utils._pytree.tree_map_only(
                        DoubleTensor, lambda x: x.inner_elem, kwargs
                    )

                    out_inner = func(*args_inner, **kwargs_inner)

                    if not isinstance(out_inner, torch.Tensor):
                        return out_inner

                    return DoubleTensor(out_inner)

            def fn(x, y):
                return torch.ops._mylib.foo(x) + y

            ins = (DoubleTensor(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([4.0, 5.0, 6.0]))
            ref_out = fn(*ins)

            compiled_fn = torch.compile(fn, backend="aot_eager")
            out = compiled_fn(*ins)