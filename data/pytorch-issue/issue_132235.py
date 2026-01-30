import torch

class TF(torch.Tensor):

    @staticmethod
    def __new__(cls, src):
        shape = src.shape
        kwargs = {}
        kwargs["strides"] = src.stride()
        kwargs["storage_offset"] = src.storage_offset()
        kwargs["device"] = src.device
        kwargs["layout"] = src.layout
        kwargs["requires_grad"] = src.requires_grad
        kwargs["dtype"] = src.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, src):
        self.src = src
    
    def __repr__(self):
        return f"{self.__class__.__name__}(src:{self.src})"

    def __tensor_flatten__(self):
        return ["src"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        src = inner_tensors["src"]
        return cls(src)
    

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        
        def f(x):
            return x.src + torch.ones(x.shape, dtype=x.dtype)

        _args = pytree.tree_map_only(cls, f, args)
        _kwargs = pytree.tree_map_only(cls, f, kwargs)
        
        raw_out = func(*_args, **_kwargs)

        out_flat, spec = pytree.tree_flatten(raw_out)
        res_out_flat = [
            cls(o) if isinstance(o, torch.Tensor) else o
            for o in out_flat
        ]
        return pytree.tree_unflatten(res_out_flat, spec)