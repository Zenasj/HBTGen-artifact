import torch

class WrapperTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad,
            strides=elem.stride(), storage_offset=elem.storage_offset())
        r.elem = elem
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        pass

x = torch.randn(2, 2)[:, 1]
print(x.storage_offset())                # prints 1
print(WrapperTensor(x).storage_offset()) # prints 0