import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

# torch.rand(3, dtype=torch.float32)
class MyModel(torch.nn.Module):
    class TwoTensor2(torch.Tensor):
        @staticmethod
        def __new__(cls, a, b):
            assert (
                a.device == b.device
                and a.layout == b.layout
                and a.requires_grad == b.requires_grad
                and a.dtype == b.dtype
            )
            shape = a.shape
            kwargs = {}
            kwargs["strides"] = a.stride()
            kwargs["storage_offset"] = a.storage_offset()
            kwargs["device"] = a.device
            kwargs["layout"] = a.layout
            kwargs["requires_grad"] = a.requires_grad
            kwargs["dtype"] = a.dtype
            kwargs["dispatch_sizes_strides_policy"] = "sizes"
            out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
            assert a.shape == b.shape
            assert a.stride() == b.stride()
            assert a.storage_offset() == b.storage_offset()
            return out

        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __repr__(self):
            a_repr = repr(self.a)
            b_repr = repr(self.b)
            return f"TwoTensor2({a_repr}, {b_repr})"

        def __tensor_flatten__(self):
            return {"a": self.a, "b": self.b}, None

        @staticmethod
        def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
            assert meta is None
            a = inner_tensors["a"]
            b = inner_tensors["b"]
            return MyModel.TwoTensor2(a, b)

        @classmethod
        def __torch_dispatch__(cls, func, types, args, kwargs):
            if kwargs is None:
                kwargs = {}
            args_a = pytree.tree_map_only(
                MyModel.TwoTensor2, lambda x: x.a, args
            )
            args_b = pytree.tree_map_only(
                MyModel.TwoTensor2, lambda x: x.b, args
            )

            kwargs_a = pytree.tree_map_only(
                MyModel.TwoTensor2, lambda x: x.a, kwargs
            )
            kwargs_b = pytree.tree_map_only(
                MyModel.TwoTensor2, lambda x: x.b, kwargs
            )

            out_a = func(*args_a, **kwargs_a)
            out_b = func(*args_b, **kwargs_b)
            assert type(out_a) == type(out_b)
            out_a_flat, spec = pytree.tree_flatten(out_a)
            out_b_flat = pytree.tree_leaves(out_b)
            out_flat = [
                MyModel.TwoTensor2(o_a, o_b) if isinstance(o_a, torch.Tensor) else o_a
                for o_a, o_b in zip(out_a_flat, out_b_flat)
            ]
            out = pytree.tree_unflatten(out_flat, spec)
            return return_and_correct_aliasing(func, args, kwargs, out)

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MyModel.TwoTensor2(x, x.clone())

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

