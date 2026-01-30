import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

class SubclassTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, a, constant):
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

    def __init__(self, a, constant):
        self.a = a
        self.constant = constant

    def __repr__(self):
        a_repr = repr(self.a)
        return f"SubclassTensor({a_repr})"

    def __tensor_flatten__(self):
        return ["a"], (self.constant,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        constant = meta[0]
        a = inner_tensors["a"]
        return SubclassTensor(a, constant)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        biggest_constant = max([x.constant for x in pytree.tree_flatten(args)[0] if isinstance(x, SubclassTensor)])
        args_a = pytree.tree_map(lambda x: x.a if isinstance(x, SubclassTensor) else x, args)
        kwargs_a = pytree.tree_map(lambda x: x.a if isinstance(x, SubclassTensor) else x, kwargs)
        out_a = func(*args_a, **kwargs_a)
        out = pytree.tree_map(lambda x: SubclassTensor(x, biggest_constant) if isinstance(x, torch.Tensor) else x, out_a)

        if func == torch.ops.aten.mul.Tensor:
            out = out + out.constant

        return return_and_correct_aliasing(func, args, kwargs, out)



def f(x):
    return x * x

x = SubclassTensor(torch.ones(2), 3)
out = f(x)

# Different metadata on the subclass should cause a recompile
x2 = SubclassTensor(torch.ones(2), 4)
# ERROR: we should recompile here, instead of reusing the graph
out2 = f(x2)
# prints SubclassTensor(tensor([4., 4.]))
# should print SubclassTensor(tensor([5., 5.]))
print(out2)